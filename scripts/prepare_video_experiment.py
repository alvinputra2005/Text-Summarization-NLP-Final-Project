"""Prepare video-separated mBART splits with a small CrossTask auxiliary sample."""

import argparse
from collections import Counter
import csv
from io import StringIO
import json
import math
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from src.youtube import fetch_youtube_captions


SPLITS = ("train", "validation", "test")
CROSSTASK_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"
CROSSTASK_SELECTION = (
    ("44047", "0bW00uLEVD0", 117.01), ("44047", "0bW00uLEVD0", 151.24),
    ("44047", "0zKC9m4hPKY", 94.96), ("44047", "0zKC9m4hPKY", 109.79),
    ("53193", "-OWofPLMkLs", 75.22), ("53193", "-OWofPLMkLs", 268.79),
    ("76400", "-QZh5_4MHPw", 48.2), ("76400", "-QZh5_4MHPw", 61.78),
    ("44047", "17ODXa4CEmI", 17.65), ("44047", "17ODXa4CEmI", 84.73),
)
FIELDS = ("id", "video_id", "topic", "start", "end", "text", "summary", "required_details", "source", "source_lang", "target_lang")


def captions(root, video_id, language):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"ID YouTube tidak valid: {video_id!r}")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{video_id}.json"
    if not path.is_file():
        with TemporaryDirectory(dir=root) as scratch:
            result = fetch_youtube_captions(f"https://www.youtube.com/watch?v={video_id}", Path(scratch), language=language)
        if result is None:
            raise ValueError(f"Subtitle {language} untuk {video_id} tidak tersedia.")
        segments, source = result
        path.write_text(json.dumps({"source": source, "segments": [[s.start, s.end, s.text] for s in segments]}, ensure_ascii=False), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["segments"]


def clip_text(segments, start, end):
    text = " ".join(line for cue_start, cue_end, line in segments if start <= (cue_start + cue_end) / 2 <= end)
    return " ".join(re.sub(r"\[(?:musik|music|tepuk tangan|applause)\]", "", text, flags=re.IGNORECASE).split())


def example(video_id, topic, start, end, text, summary, required, source, lang, number):
    if not (0 <= start < end and math.isfinite(end)) or not text or not summary:
        raise ValueError(f"Contoh kosong atau interval tidak valid: {video_id} {start}–{end}")
    return dict(zip(FIELDS, (
        f"{video_id}:{number}", video_id, topic, start, end, text, summary,
        json.dumps(required, ensure_ascii=False), source, lang, lang,
    ), strict=True))


def natural_rows(manifest, raw_dir):
    result = {split: [] for split in SPLITS}
    videos = set()
    for split in SPLITS:
        for video in manifest[split]:
            video_id = video["video_id"]
            if video_id in videos:
                raise ValueError(f"Video muncul lebih dari sekali pada manifest: {video_id}")
            videos.add(video_id)
            segments = captions(raw_dir, video_id, "id")
            for number, (start, end, summary, details) in enumerate(video["clips"], 1):
                result[split].append(example(video_id, video["topic"], start, end, clip_text(segments, start, end),
                                             summary, details, "python_youtube", "id_ID", number))
    return result


def cross_rows(root):
    root.mkdir(parents=True, exist_ok=True)
    release = root / "crosstask_release.zip"
    if not release.exists():
        with urlopen(CROSSTASK_URL, timeout=30) as response:
            release.write_bytes(response.read())
    with ZipFile(release) as archive:
        def read(name):
            return archive.read(f"crosstask_release/{name}").decode("utf-8-sig")

        blocks = re.split(r"\r?\n\s*\r?\n", read("tasks_primary.txt").strip())
        tasks = {lines[0]: (lines[1], lines[4].split(",")) for block in blocks if len(lines := block.splitlines()) == 5}
        excluded = {video_id for _, video_id, _ in csv.reader(StringIO(read("videos_val.csv")))}
        for task, video_id, anchor in CROSSTASK_SELECTION:
            if video_id in excluded:
                raise ValueError(f"Video validasi CrossTask ditemukan pada data latih: {video_id}")
            annotations = csv.reader(StringIO(read(f"annotations/{task}_{video_id}.csv")))
            step, start, end = next(row for row in annotations if float(row[1]) == anchor)
            title, labels = tasks[task]
            label = labels[int(step) - 1].strip()
            segments = captions(root, video_id, "en")
            # Subtitle cues may straddle an annotated visual boundary.
            text = clip_text(segments, float(start) - 4, float(end) + 4)
            terms = [word for word in label.split() if word not in {"in", "into", "from", "on", "off"}]
            if not all(re.search(rf"\b{re.escape(word)}\w*\b", text, re.IGNORECASE) for word in terms):
                raise ValueError(f"Label dan subtitle CrossTask tidak cocok: {video_id} {start} {label}")
            yield example(video_id, title, float(start), float(end), text, label.capitalize() + ".",
                          terms, "crosstask", "en_XX", start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("samples/coding_video_segments.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/coding_video_mix"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = natural_rows(manifest, Path("data/raw/coding_video"))
    rows["train"].extend(cross_rows(Path("data/raw/crosstask")))
    with Path("data/processed/coding_pilot/train.csv").open(encoding="utf-8", newline="") as file:
        for row in random.Random(42).sample(list(csv.DictReader(file)), 28):
            rows["train"].append({key: row[key] for key in ("id", "topic", "text", "summary", "source")} |
                                 {"source_lang": "id_ID", "target_lang": "id_ID"})

    seen = {}
    for split in SPLITS:
        for row in rows[split]:
            key = " ".join(row["text"].casefold().split())
            if key in seen and seen[key] != split:
                raise ValueError(f"Transkrip yang sama ditemukan di dua split: {row['id']}")
            seen[key] = split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (args.output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows[split])
        print(split, len(rows[split]), dict(Counter(row["source"] for row in rows[split])))


if __name__ == "__main__":
    main()
