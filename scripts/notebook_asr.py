"""Audio and evaluation routines shared by the two Whisper notebooks."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from src.evaluation import normalize, speech_metrics, word_errors


SAMPLE_RATE = 16_000


def read_audio(path: Path) -> np.ndarray:
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ], capture_output=True, check=True, timeout=60)
    signal = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not 0 < signal.size <= 30 * SAMPLE_RATE or not np.isfinite(signal).all() or not signal.any():
        raise ValueError(f"Klip harus berisi audio berdurasi lebih dari 0 dan paling lama 30 detik: {path}")
    return signal


def read_manifest(path: Path, media_column: str) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if rows.empty or rows["id"].duplicated().any():
        raise ValueError(f"Manifest kosong atau ID ganda: {path}")
    rows["media"] = rows[media_column].map(lambda name: (path.parent / name).resolve())
    invalid = [str(row.media) for row in rows.itertuples()
               if not row.media.is_file() or not normalize(row.transcript)]
    if invalid:
        raise ValueError(f"Media atau transkrip tidak valid: {invalid[:3]}")
    return rows


def encode_clips(rows, processor, model):
    clips = []
    for row in rows.to_dict("records"):
        audio = read_audio(row["media"])
        features = processor(audio, sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt")
        labels = processor.tokenizer(row["transcript"], return_tensors="pt").input_ids
        if labels[0, 0].item() == model.config.decoder_start_token_id:
            labels = labels[:, 1:]
        if labels.shape[1] > model.config.max_target_positions:
            raise ValueError(f"Transkrip melewati batas posisi decoder: {row['id']}")
        clips.append((row, features, labels))
    return clips


@torch.inference_mode()
def evaluate_asr(model, processor, clips, device):
    model.eval()
    predictions = []
    for row, features, _ in clips:
        generated = model.generate(
            **features.to(device), language="indonesian", task="transcribe",
            num_beams=1, do_sample=False, max_new_tokens=128,
            return_dict_in_generate=True, output_scores=True,
        )
        prediction = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        token_scores = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)[0]
        generated_tokens = generated.sequences[0, -token_scores.numel():]
        token_scores = token_scores[generated_tokens < processor.tokenizer.eos_token_id]
        logprob = float(token_scores.mean()) if token_scores.numel() else float("nan")
        reference = row["transcript"]
        predictions.append({
            "id": row["id"], "reference": reference, "prediction": prediction,
            "speaker_id": row.get("speaker_id", ""), "video_id": row.get("video_id", ""),
            "mean_logprob": logprob, **speech_metrics([reference], [prediction]), **word_errors(reference, prediction),
        })
    frame = pd.DataFrame(predictions)
    return speech_metrics(frame.reference.tolist(), frame.prediction.tolist()), frame


def risk_roc(results: pd.DataFrame, threshold: float = 0.30):
    """Use independent reference WER labels and rank by decoder confidence."""
    usable = results.loc[np.isfinite(results["mean_logprob"])].copy()
    usable["error"] = usable["wer"] >= threshold
    positives = int(usable["error"].sum())
    negatives = len(usable) - positives
    if not positives or not negatives:
        return None
    counts = usable.groupby("mean_logprob", sort=True)["error"].agg(["sum", "size"])
    tp = np.r_[0, counts["sum"].cumsum().to_numpy() / positives]
    fp = np.r_[0, (counts["size"] - counts["sum"]).cumsum().to_numpy() / negatives]
    auc = float(np.sum(np.diff(fp) * (tp[:-1] + tp[1:]) / 2))
    return fp, tp, auc, len(usable)
