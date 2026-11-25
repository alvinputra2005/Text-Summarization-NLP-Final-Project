"""Compare whole-video summaries with the same Whisper text and mBART checkpoint."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

from scripts.notebook_multimodal import load_transcript
from scripts.notebook_video_summary import score_video
from src.contextual_cleaning import contextual_parts, refine_context
from src.evaluation import exact_code_details, normalize
from src.gemini_preprocessing import GeminiError, gemini_config
from src.segmentation import Segment
from src.video_context import (
    MAX_POINTS, capture_parts, draft, enrich, probe_duration, read_with_gemini, speech_drafts,
    stable_actions, validated_observations, video_parts,
)
from src.video_ocr import download_video, tesseract_path
from src.video_outline import BOILERPLATE
from src.summarization import guard_summary
from src.youtube import youtube_watch_url


def select_transcript(cues, whisper, video, url):
    if cues and max(cue.end - cue.start for cue in cues) <= 45:
        return cues, whisper, None
    reason = "Tidak ada segmen" if not cues else "Rentang satu segmen melebihi 45 detik"
    fallback, fallback_model = load_transcript(video, None, url)
    return fallback, fallback_model, {"model": whisper, "reason": reason, "segments": len(cues)}


def capture_transcript(video, transcript, url, whisper_checkpoint=None):
    started = perf_counter()
    duration = probe_duration(video)
    retry = None
    if whisper_checkpoint and not transcript:
        import torch
        from transformers import pipeline

        torch.set_num_threads(4)
        asr = pipeline("automatic-speech-recognition", model=str(whisper_checkpoint),
                       chunk_length_s=30, stride_length_s=(4, 4), device=-1)
        result = asr(str(video), return_timestamps=True,
                     generate_kwargs={"language": "indonesian", "task": "transcribe"})
        cues = [
            Segment(start, end if end is not None else duration, item["text"])
            for item in result["chunks"] for start, end in (item["timestamp"],)
            if start is not None and start < (end if end is not None else duration) and item["text"].strip()
        ]
        whisper = f"Whisper sesudah pelatihan tambahan ({whisper_checkpoint.parent.name})"
        del asr
        cues, whisper, retry = select_transcript(cues, whisper, video, url)
    else:
        cues, whisper = load_transcript(video, transcript, url)
    return {"url": url, "video_path": str(video.resolve()), "duration": duration,
            "whisper": whisper, "cues": [asdict(cue) for cue in cues],
            "transcript_seconds": perf_counter() - started, "cached_transcript": bool(transcript),
            "transcription_retry": retry}


def clean_transcript(state, part_seconds):
    started = perf_counter()
    state["parts"] = video_parts([Segment(**cue) for cue in state["cues"]], state["duration"], part_seconds)
    state["cleaning_seconds"] = perf_counter() - started
    return state


def extract_visual(state, roi):
    started = perf_counter()
    engine = tesseract_path()
    capture_parts(Path(state["video_path"]), state["parts"], roi, engine)
    state["ocr_seconds"] = perf_counter() - started
    state["ocr_available"] = bool(engine)
    state["actions"] = stable_actions(state["parts"])
    return state


def enrich_with_gemini(state, max_requests):
    started = perf_counter()
    state.pop("context_cleaning", None)
    reserved = int(max_requests > 1)
    state["gemini"] = read_with_gemini(state["parts"], max_requests - reserved)
    if reserved and not state["gemini"].get("failed"):
        try:
            config = gemini_config()
        except GeminiError as error:
            state["context_cleaning"] = {"status": str(error), "requests": 0, "changes": []}
        else:
            state["context_cleaning"] = refine_context(state["parts"], state["gemini"]["verified"], config)
    state["gemini"]["seconds"] = perf_counter() - started
    for part in state["parts"]:
        for frame in part["frames"]:
            frame.pop("crop", None)
    return state


def prepare(video, transcript, url, part_seconds, roi, gemini_max_requests):
    state = capture_transcript(video, transcript, url)
    clean_transcript(state, part_seconds)
    extract_visual(state, roi)
    return enrich_with_gemini(state, gemini_max_requests)


def render(drafts, runner):
    results, seen = [], set()
    countdown = MAX_POINTS
    for item in drafts:
        if countdown == 0:
            break
        text = item["text"]
        source = f"{text} Ucapan sumber: {item['quote']}" if item.get("quote") else text
        candidate = runner.generate(runner.encode(source)[0])
        text = guard_summary(text, candidate, required_clause=item["required"]).text
        plain = normalize(text)
        if plain and plain not in seen and not BOILERPLATE.search(text):
            results.append({**item, "summary": text})
            seen.add(plain)
            countdown -= 1
    return results


def evaluate(rows, reference):
    facts = reference.get("facts") if reference else None
    if not facts:
        return None
    scores = score_video(rows, facts)
    details = reference.get("code_details", [])
    scores["detail_hits"] = exact_code_details(details, [row["summary"] for row in rows])
    scores["detail_recall"] = sum(item["present"] for item in scores["detail_hits"]) / len(details) if details else None
    return scores


def summarize(prepared, reference, mbart_weights=None, *, device="cpu", download=False):
    import torch
    from src.seq2seq_models import load_runner

    torch.set_num_threads(4)
    dtype = torch.bfloat16
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float32
    runner = load_runner("mbart", dtype=dtype, device=device, download=download)
    if mbart_weights:
        from scripts.notebook_training import load_mbart_weights

        load_mbart_weights(runner, mbart_weights)
    drafts = speech_drafts(prepared["parts"])
    variants = {"transcript": drafts, "ocr": enrich(drafts, prepared, [])}
    context_report = prepared.get("context_cleaning", {})
    if prepared["gemini"]["verified"] or context_report.get("changes"):
        context = {**prepared, "parts": contextual_parts(prepared["parts"], context_report)}
        variants["gemini"] = enrich(speech_drafts(context["parts"]), context, prepared["gemini"]["verified"])
    outputs = {}
    for name, items in variants.items():
        started = perf_counter()
        rows = render(items, runner)
        outputs[name] = {"rows": rows, "seconds": perf_counter() - started, "scores": evaluate(rows, reference)}
    precision = "BF16" if dtype == torch.bfloat16 else "FP32"
    return {"variants": outputs, "checkpoint": runner.identity, "precision": f"{precision} · {device}",
            "reference": reference["title"] if reference else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "transcribe", "clean", "ocr", "gemini", "summarize"))
    parser.add_argument("--url", default="")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--whisper-checkpoint", type=Path)
    parser.add_argument("--mbart-weights", type=Path)
    parser.add_argument("--reference", type=Path, help="JSON acuan manual untuk video yang sedang diuji.")
    parser.add_argument("--download", action="store_true", help="Unduh checkpoint mBART yang belum tersimpan.")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--part-seconds", type=int, default=55)
    parser.add_argument("--roi", type=float, nargs=4, default=(0.225, 0.115, 0.61, 0.90))
    parser.add_argument("--gemini-max-requests", type=int, default=4)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.part_seconds < 20 or options.gemini_max_requests < 0:
        parser.error("Durasi bagian minimal 20 detik. Batas panggilan harus nonnegatif.")
    x0, y0, x1, y1 = options.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error("Koordinat crop harus berurutan dan berada pada rentang 0 sampai 1.")
    destination = options.prepared
    if options.stage in {"prepare", "transcribe"}:
        if not (options.url or options.video):
            parser.error("Siapkan tautan YouTube atau MP4 lokal.")
        url = youtube_watch_url(options.url) if options.url else ""
        video = options.video or download_video(url, options.prepared.parent / "media")
        result = capture_transcript(video, options.transcript, url, options.whisper_checkpoint)
        if options.stage == "prepare":
            clean_transcript(result, options.part_seconds)
            extract_visual(result, options.roi)
            enrich_with_gemini(result, options.gemini_max_requests)
    else:
        result = json.loads(options.prepared.read_text(encoding="utf-8"))
        if options.stage == "clean":
            clean_transcript(result, options.part_seconds)
        elif options.stage == "ocr":
            extract_visual(result, options.roi)
        elif options.stage == "gemini":
            enrich_with_gemini(result, options.gemini_max_requests)
        else:
            if not options.output:
                parser.error("Tahap summarize membutuhkan --output.")
            destination = options.output
            video_id = parse_qs(urlsplit(result["url"]).query).get("v", [""])[0]
            manifest = Path("samples/notebook_video_holdout.json")
            references = json.loads(manifest.read_text(encoding="utf-8"))
            reference = next((item for item in references if item["id"] == video_id), None)
            if options.reference:
                reference = json.loads(options.reference.read_text(encoding="utf-8"))
                if not isinstance(reference, dict) or not reference.get("title") or not reference.get("facts"):
                    parser.error("Acuan harus berisi title dan facts seperti salah satu entri manifest video.")
            result = summarize(result, reference, options.mbart_weights, device=options.device,
                               download=options.download)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
