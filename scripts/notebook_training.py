"""Run the notebook's mixed Whisper and staged mBART experiments."""

import argparse
import json
from pathlib import Path
import random
from statistics import mean


ROOT = Path("data/processed")
SEED = 42
SPLITS = ("train", "validation", "test")
AUDIO = {
    "LUMINA": ("lumina_split", SPLITS, "video_path", "speaker_id"),
    "NSS-ID": ("nss-id", ("train", "val", "test"), "audio_path", "speaker_id"),
    "Audio coding": ("whisper_video", SPLITS, "audio_path", "video_id"),
}


def save_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audio_plan(profile):
    from scripts.notebook_asr import read_manifest

    caps = (2, 2, 2) if profile == "smoke" else (16, 8, 12)
    plan = {}
    holdout_ids = {row["id"] for row in json.loads(Path("samples/notebook_video_holdout.json").read_text("utf-8"))}
    for name, (folder, aliases, media, group) in AUDIO.items():
        selected = {}
        seen_groups = set()
        seen_media = set()
        for split, alias, cap in zip(SPLITS, aliases, caps, strict=True):
            frame = read_manifest(ROOT / folder / f"{alias}.csv", media)
            groups = set(frame[group])
            paths = set(frame["media"])
            if "" in groups or seen_groups & groups or seen_media & paths or (
                group == "video_id" and groups & holdout_ids
            ):
                raise ValueError(f"Pembagian media, penutur atau video bermasalah pada {name}/{split}.")
            seen_groups |= groups
            seen_media |= paths
            selected[split] = frame.sort_values("id").sample(min(cap, len(frame)), random_state=SEED)
        plan[name] = selected
    return plan


def train_whisper(profile, output, device_name):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
    from scripts.notebook_asr import encode_clips, evaluate_asr

    device = torch.device(device_name)
    local = Path("data/raw/models/whisper-base-hf")
    model_id = str(local) if (local / "model.safetensors").is_file() else "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_id, language="id", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    plan = audio_plan(profile)
    encoded = {name: {split: encode_clips(frame, processor, model) for split, frame in splits.items()}
               for name, splits in plan.items()}
    training = [clip for splits in encoded.values() for clip in splits["train"]]
    model.requires_grad_(False)
    last = model.model.decoder.layers[-1]
    for attention in (last.self_attn, last.encoder_attn):
        attention.q_proj.requires_grad_(True)
        attention.v_proj.requires_grad_(True)
    params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-5, weight_decay=0.01, foreach=False)
    epochs = 1 if profile == "smoke" else 5
    updates = epochs * len(training)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * updates), updates)
    baseline = {name: evaluate_asr(model, processor, splits["test"], device)[0] for name, splits in encoded.items()}
    history, best_key, best_weights = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = random.Random(SEED + epoch).sample(training, len(training))
        losses = []
        for _, features, labels in shuffled:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{key: value.to(device) for key, value in features.items()},
                         labels=labels.to(device), use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss Whisper tidak finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        valid = {name: evaluate_asr(model, processor, splits["validation"], device)[0]
                 for name, splits in encoded.items()}
        key = (mean(score["wer"] for score in valid.values()), mean(score["cer"] for score in valid.values()))
        history.append({"epoch": epoch, "loss": mean(losses), "macro_wer": key[0], "macro_cer": key[1],
                        "domains": valid, "lr_end": optimizer.param_groups[0]["lr"]})
        print(f"Whisper epoch {epoch} loss={mean(losses):.3f} macro-WER validasi={key[0]:.3f}", flush=True)
        if best_key is None or key < best_key:
            best_key, best_epoch = key, epoch
            best_weights = snapshot(params)
        if epoch - best_epoch >= 2:
            break
    restore(params, best_weights)
    results = {name: evaluate_asr(model, processor, splits["test"], device)[0] for name, splits in encoded.items()}
    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    return {"mode": profile, "base": "openai/whisper-base", "checkpoint": str(checkpoint), "device": device_name,
            "learning_rate": 1e-5, "max_epochs": epochs, "best_epoch": best_epoch, "history": history,
            "splits": {name: {split: len(rows) for split, rows in parts.items()} for name, parts in plan.items()},
            "test": {name: {"base": baseline[name], "adapted": results[name]} for name in encoded}}


def snapshot(params):
    return {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in params.items()}


def restore(params, weights):
    import torch

    if set(params) != set(weights):
        raise ValueError("Parameter checkpoint tidak cocok dengan model.")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(weights[name])


def mbart_parameters(model):
    prefix = f"model.decoder.layers.{len(model.model.decoder.layers) - 1}.encoder_attn."
    names = {prefix + "q_proj.weight", prefix + "v_proj.weight"}
    return {name: parameter for name, parameter in model.named_parameters()
            if name in names}


def load_mbart_weights(runner, path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base"] != runner.spec.hub or payload["revision"] != runner.spec.revision:
        raise ValueError("Checkpoint memakai revisi mBART yang berbeda.")
    restore(mbart_parameters(runner.model), payload["params"])
    runner.checkpoint = f"{payload['strategy']} · {payload['mode']}"


def fit_mbart(runner, params, training, validation, domain, epochs):
    import torch
    from scripts.compare_seq2seq import evaluate, score_key

    master = {name: torch.nn.Parameter(parameter.detach().float().clone()) for name, parameter in params.items()}
    optimizer = torch.optim.AdamW(master.values(), lr=2e-5, weight_decay=0.01, foreach=False)
    history, best_key, best_weights = [], None, None
    for epoch in range(1, epochs + 1):
        runner.model.train()
        rows = random.Random(SEED + epoch).sample(training, len(training))
        losses = []
        for start in range(0, len(rows), 8):
            group = rows[start:start + 8]
            optimizer.zero_grad(set_to_none=True)
            runner.model.zero_grad(set_to_none=True)
            for _, batch, _ in group:
                batch = {key: tensor.to(runner.model.device) for key, tensor in batch.items()}
                loss = runner.model(**batch, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Loss mBART tidak finite.")
                (loss / len(group)).backward()
                losses.append(loss.item())
            for name, parameter in params.items():
                if parameter.grad is not None:
                    master[name].grad = parameter.grad.detach().float()
            torch.nn.utils.clip_grad_norm_(master.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            restore(params, master)
        valid, _ = evaluate(runner, validation)
        key = score_key(valid, domain)
        history.append({"epoch": epoch, "loss": mean(losses), "validation": valid})
        print(f"mBART {domain} epoch {epoch} loss={mean(losses):.3f} ROUGE-L validasi={valid['rougeL']:.3f}", flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_weights = snapshot(params)
        if epoch - best_epoch >= 2:
            break
    restore(params, best_weights)
    return {"dataset": domain, "history": history, "best_epoch": best_epoch}


def train_mbart(profile, output, device):
    import torch
    from scripts.compare_seq2seq import encode_rows, evaluate, experiment_rows, score_key
    from src.seq2seq_models import load_runner

    runner = load_runner("mbart", device=device, dtype=torch.bfloat16, source_tokens=256, target_tokens=72)
    raw = {name: experiment_rows(name, profile) for name in ("indosum", "coding_video_mix")}
    if profile == "smoke":
        raw = {name: {role: rows[:2] for role, rows in splits.items()} for name, splits in raw.items()}
    encoded = {name: {role: encode_rows(runner, rows) for role, rows in splits.items()}
               for name, splits in raw.items()}
    coding = encoded["coding_video_mix"]
    model = runner.model
    model.requires_grad_(False)
    params = mbart_parameters(model)
    if len(params) != 2:
        raise RuntimeError("Dua proyeksi cross-attention decoder terakhir diperlukan.")
    for parameter in params.values():
        parameter.requires_grad_(True)
    original = snapshot(params)
    epochs = 1 if profile == "smoke" else 5
    variants, weights = {}, {}
    for name, stages in (
        ("base", ()), ("coding", ("coding_video_mix",)), ("indosum_coding", ("indosum", "coding_video_mix")),
    ):
        restore(params, original)
        history = [fit_mbart(runner, params, encoded[domain]["train"], encoded[domain]["validation"], domain, epochs)
                   for domain in stages]
        validation = history[-1]["history"][history[-1]["best_epoch"] - 1]["validation"] if history else evaluate(
            runner, coding["validation"])[0]
        variants[name] = {"validation": validation, "stages": history}
        weights[name] = snapshot(params)
    winner = max(variants, key=lambda name: score_key(variants[name]["validation"], "coding_video_mix"))
    for name, variant in variants.items():
        restore(params, weights[name])
        variant["test"] = evaluate(runner, coding["test"])[0]
    checkpoint = output / "selected.pt"
    torch.save({"base": runner.spec.hub, "revision": runner.spec.revision, "strategy": winner,
                "mode": profile, "params": weights[winner]}, checkpoint)
    return {"mode": profile, "base": runner.spec.hub, "revision": runner.spec.revision,
            "device": device, "checkpoint": str(checkpoint), "winner": winner, "variants": variants,
            "learning_rate": 2e-5, "batch": 1, "accumulation": 8, "max_epochs": epochs,
            "parameter_count": sum(parameter.numel() for parameter in params.values()),
            "splits": {name: {split: len(rows) for split, rows in parts.items()} for name, parts in raw.items()}}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("whisper", "mbart"))
    parser.add_argument("--profile", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    fit = train_whisper if args.task == "whisper" else train_mbart
    save_report(args.output / "report.json", fit(args.profile, args.output, args.device))


if __name__ == "__main__":
    main()
