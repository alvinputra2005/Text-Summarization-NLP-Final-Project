"""Evaluate Whisper against a transcript manifest or the guide against a timestamped reference."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from src.evaluation import normalize, speech_metrics
from src.transcription import WHISPER_MODEL, transcribe, whisper_backend


def evaluate_whisper(manifest, limit):
    with manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not 1 <= limit <= len(rows):
        raise ValueError(f"Gunakan 1 sampai {len(rows)} contoh dari {manifest}.")
    sample = random.Random(42).sample(rows, limit)
    results = []
    for item in sample:
        media = manifest.parent / (item.get("audio_path") or item["video_path"])
        started = perf_counter()
        prediction = " ".join(segment.transcript for segment in transcribe(media)).strip()
        results.append({
            "id": item["id"], "reference": item["transcript"], "prediction": prediction,
            "transcription_seconds": perf_counter() - started,
        })
        print(f"Whisper {len(results)}/{limit}", flush=True)
    return {
        "manifest": str(manifest),
        "whisper": WHISPER_MODEL,
        "whisper_backend": whisper_backend(),
        "language": "id",
        "seed": 42,
        "samples": limit,
        "transcription_seconds": sum(item["transcription_seconds"] for item in results),
        **speech_metrics([item["reference"] for item in results], [item["prediction"] for item in results]),
        "predictions": results,
    }


def overlap_ratio(reference, predicted):
    overlap = min(reference["end"], predicted.end) - max(reference["start"], predicted.start)
    span = max(reference["end"], predicted.end) - min(reference["start"], predicted.start)
    return max(0, overlap) / span


def evaluate_tutorial(video, reference_path):
    from rouge_score import rouge_scorer

    from src.pipeline import build_guide

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["steps"]
    if not expected or any(not 0 <= item["start"] < item["end"] for item in expected):
        raise ValueError("Referensi tutorial harus berisi langkah dengan interval waktu yang valid.")
    guide = build_guide(video.read_bytes(), lambda _, message: print(message, flush=True))
    candidates = sorted(
        ((overlap_ratio(item, step), left, right)
         for left, item in enumerate(expected) for right, step in enumerate(guide.steps)),
        reverse=True,
    )
    pairs = {}
    used_predictions = set()
    for score, left, right in candidates:
        if score >= 0.5 and left not in pairs and right not in used_predictions:
            pairs[left] = (guide.steps[right], score)
            used_predictions.add(right)

    score_types = ("rouge1", "rouge2", "rougeL")
    scorer = rouge_scorer.RougeScorer(score_types)
    totals = Counter()
    details = []
    for index, item in enumerate(expected):
        step, score = pairs.get(index, (None, 0.0))
        transcript = step.transcript if step else ""
        summary = step.summary if step else ""
        term_hits = {}
        for field, value in (("transcript", transcript), ("summary", summary)):
            term_hits[field] = {
                term: f" {normalize(term)} " in f" {normalize(value)} " for term in item["terms"]
            }
            totals[f"{field}_term_hits"] += sum(term_hits[field].values())
        totals["terms"] += len(item["terms"])
        totals.update({name: value.fmeasure for name, value in scorer.score(item["summary"], summary).items()})
        seek_start = int(step.start) if step else None
        details.append({
            "reference": item,
            "prediction": asdict(step) if step else None,
            "temporal_iou": score,
            "seek_start": seek_start,
            "timestamp_hit": seek_start is not None and item["start"] <= seek_start < item["end"],
            "term_hits": term_hits,
        })

    return {
        "fixture": reference["kind"],
        "video": str(video),
        "whisper": guide.transcription_model,
        "whisper_backend": guide.transcription_backend,
        "processing_seconds": guide.processing_seconds,
        "timings": asdict(guide.timings) if guide.timings is not None else None,
        "mbart": guide.model,
        "reference_steps": len(expected),
        "predicted_steps": len(guide.steps),
        "matched_steps": len(pairs),
        "timestamp_hits": sum(item["timestamp_hit"] for item in details),
        "mean_temporal_iou": mean(item["temporal_iou"] for item in details),
        "terms": totals["terms"],
        "transcript_term_hits": totals["transcript_term_hits"],
        "summary_term_hits": totals["summary_term_hits"],
        **{name: totals[name] / len(expected) for name in score_types},
        **speech_metrics(
            [" ".join(item["transcript"] for item in expected)],
            [" ".join(step.transcript for step in guide.steps)],
        ),
        "steps": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    whisper = subparsers.add_parser("whisper")
    whisper.add_argument("manifest", type=Path)
    whisper.add_argument("--limit", type=int, default=8)
    tutorial = subparsers.add_parser("tutorial")
    tutorial.add_argument("video", type=Path)
    tutorial.add_argument("reference", type=Path)
    for subparser in (whisper, tutorial):
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Berkas laporan sudah ada: {args.output}")

    if args.task == "whisper":
        report = evaluate_whisper(args.manifest, args.limit)
    else:
        report = evaluate_tutorial(args.video, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({key: val for key, val in report.items() if key not in {"steps", "predictions"}}, indent=2))


if __name__ == "__main__":
    main()
