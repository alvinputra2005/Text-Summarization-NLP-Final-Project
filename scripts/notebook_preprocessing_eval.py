"""Evaluate source selection with public transcript caches and controlled fixtures."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from scripts.notebook_video_summary import match_terms
from src.evaluation import exact_code_details
from src.pipeline import select_guide_sources
from src.segmentation import Segment
from src.step_sources import prepare_step_sources
from src.transcript_quality import coding_transcript_status
from src.video_context import scan_context, visual_branch_sources
from src.video_ocr import tesseract_path
from src.video_outline import limit_tutorial_sources


def render_screen(lines, directory):
    image = Image.new("RGB", (1280, 720), "#1b2029")
    for font_name in ("C:/Windows/Fonts/cour.ttf", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(font_name, 36)
            break
        except OSError:
            continue
    else:
        raise FileNotFoundError("Font Courier New atau DejaVu Sans Mono diperlukan untuk klip uji.")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        draw.text((725 + 18 * indent, 170 + 52 * index), line.lstrip(), fill="white", font=font)
    png, clip = directory / "layar.png", directory / "layar.mp4"
    image.save(png)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(png),
        "-t", "15", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(clip),
    ], check=True, timeout=60)
    return clip


def measure(case, sources):
    points = [{"start": item.segment.start, "end": item.segment.end, "text": item.text,
               "source": item.segment.transcript} for item in sources if item.text]
    facts = []
    for expected in case["facts"]:
        hits = [point for point in points if match_terms(point["text"], expected["terms"])]
        facts.append({"fact": expected["summary"], "covered": bool(hits), "aligned": any(
            point["start"] < expected["end"] and point["end"] > expected["start"] for point in hits
        )})
    return {"points": points, "facts": facts,
            "details": exact_code_details(case.get("code_details", []), [point["text"] for point in points])}


def evaluate(case, output_dir):
    segments = [Segment(**item) for item in case["transcript"]]
    status = coding_transcript_status(segments)
    baseline = limit_tutorial_sources(prepare_step_sources(segments)) if status == "ready" else []
    revised = select_guide_sources(segments) if segments else []
    visual = None
    if "screen_code" in case:
        clip = render_screen(case["screen_code"], output_dir)
        context = scan_context(clip, segments, tesseract_path(), None, None, screen_only=True)
        revised = visual_branch_sources(segments, context["parts"], [], screen_only=True)
        single_frame = [{**part, "frames": part["frames"][:1]} for part in context["parts"]]
        visual = {"frames": [{"second": frame["second"], "lines": frame["lines"]}
                             for part in context["parts"] for frame in part["frames"]],
                  "single_frame_points": len(visual_branch_sources(segments, single_frame, [], screen_only=True))}
    return {"id": case["id"], "title": case["title"], "kind": case["kind"], "role": case["role"],
            "url": case.get("video_url"), "transcript_model": case.get("whisper_model"),
            "cues": len(segments), "status": status, "visual": visual,
            "baseline": measure(case, baseline), "revised": measure(case, revised)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/notebooks/heldout-small"))
    args = parser.parse_args()
    public = json.loads(Path("samples/notebook_video_holdout.json").read_text(encoding="utf-8"))
    synthetic = json.loads(Path("samples/preprocessing_regression.json").read_text(encoding="utf-8"))
    measured, missing = [], []
    for case in public:
        cache = args.cache_dir / f"{case['id']}.json"
        if not cache.is_file():
            missing.append(case["id"])
            continue
        capture = json.loads(cache.read_text(encoding="utf-8"))
        measured.append(evaluate({**case, **capture, "kind": "Video publik"}, args.output.parent))
    for case in synthetic:
        directory = args.output.parent / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        measured.append(evaluate({**case, "kind": "Sintetis", "role": "Uji regresi"}, directory))
    report = {"cases": measured, "missing_public": missing}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(measured)} sampel selesai, {len(missing)} cache publik belum tersedia.")


if __name__ == "__main__":
    main()
