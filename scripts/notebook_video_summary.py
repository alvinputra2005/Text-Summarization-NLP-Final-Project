"""Compare transcript selection policies on whole, held-out tutorial videos."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from src.evaluation import normalize, summary_metrics
from src.segmentation import Segment
from src.step_sources import prepare_step_sources
from src.transcript_quality import is_readable_summary
from src.video_outline import BOILERPLATE, anchors, select_across_video


def render_sources(sources, tokenizer, model, *, preserve_anchors):
    from src.summarization import summarize

    rows = []
    for source in sources:
        if not source.text:
            continue
        result = summarize(source.text, tokenizer, model, required_clause=source.required_clause)
        text = result.text
        use_excerpt = result.is_excerpt
        if not is_readable_summary(text) or (preserve_anchors and anchors(source.text) != anchors(text)):
            text, use_excerpt = source.text, True
        rows.append({
            **asdict(source.segment), "input": source.text, "summary": text,
            "origin": "cuplikan sumber" if use_excerpt else "mBART",
        })
    return rows


def match_terms(summary, required):
    text = f" {normalize(summary)} "
    return all(any(f" {normalize(term)} " in text for term in alternatives) for alternatives in required)


def score_video(rows, facts):
    inventory = []
    for fact in facts:
        hits = [row for row in rows if match_terms(row["summary"], fact["terms"])]
        aligned = any(min(row["end"], fact["end"]) > max(row["start"], fact["start"]) for row in hits)
        inventory.append({"fact": fact["summary"], "covered": bool(hits), "aligned": aligned})
    return {
        "points": len(rows),
        "boilerplate": sum(bool(BOILERPLATE.search(row["summary"])) for row in rows),
        "coverage": sum(row["covered"] for row in inventory) / len(facts),
        "aligned_coverage": sum(row["aligned"] for row in inventory) / len(facts),
        **summary_metrics(" ".join(fact["summary"] for fact in facts), " ".join(row["summary"] for row in rows)),
        "inventory": inventory,
    }


def transcribe_video(video_id, whisper_model):
    import os

    from src.transcription import transcribe
    from src.youtube import download_youtube_audio, youtube_watch_url

    os.environ["WHISPER_MODEL_ID"] = whisper_model
    os.environ["WHISPER_BEAM_SIZE"] = "3"
    os.environ["WHISPER_BACKEND"] = "faster"
    url = youtube_watch_url(f"https://youtu.be/{video_id}")
    with TemporaryDirectory() as scratch:
        audio = download_youtube_audio(url, Path(scratch))
        start = perf_counter()
        cues = transcribe(audio, retry_noisy=True)
    return {"transcript": [asdict(cue) for cue in cues], "seconds": perf_counter() - start,
            "whisper_model": whisper_model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("samples/notebook_video_holdout.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/notebooks/heldout-small"))
    parser.add_argument("--whisper-model", default="data/raw/models/faster-whisper-small")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    import torch
    from src.summarization import PUBLIC_MODEL, load_mbart

    torch.set_num_threads(4)
    references = json.loads(options.manifest.read_text(encoding="utf-8"))
    options.cache_dir.mkdir(parents=True, exist_ok=True)
    captures = {}
    for video in references:
        video_id = video["id"]
        cache = options.cache_dir / f"{video_id}.json"
        capture = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        if capture.get("whisper_model") != options.whisper_model:
            capture = transcribe_video(video_id, options.whisper_model)
            cache.write_text(json.dumps(capture, ensure_ascii=False), encoding="utf-8")
        captures[video_id] = capture
    tokenizer, model = load_mbart(PUBLIC_MODEL)
    report = []
    for video in references:
        capture = captures[video["id"]]
        cues = [Segment(**row) for row in capture["transcript"]]
        comparison = {}
        for label, sources, check in (
            ("local", prepare_step_sources(cues), False),
            ("coverage", select_across_video(cues), True),
        ):
            started = perf_counter()
            rows = render_sources(sources, tokenizer, model, preserve_anchors=check)
            comparison[label] = {"rows": rows, "scores": score_video(rows, video["facts"]),
                                 "seconds": perf_counter() - started}
        report.append({
                "id": video["id"], "title": video["title"], "role": video["role"], "whisper": capture["whisper_model"], "mbart": PUBLIC_MODEL,
            "transcription_seconds": capture["seconds"], "segments": len(cues), "comparison": comparison,
        })
        print(f"{video['title']}: {comparison['local']['scores']['coverage']:.0%} / "
              f"{comparison['coverage']['scores']['coverage']:.0%}", flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
