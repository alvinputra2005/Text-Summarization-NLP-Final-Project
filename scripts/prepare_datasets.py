"""Prepare local CSV splits and audio/clip manifests from the downloaded datasets."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile
from openpyxl import load_workbook
import pyarrow.parquet as pq


RAW = Path("data/raw")
PREPARED = Path("data/processed")
SPLITS = (("test", "test"), ("validation", "dev"), ("train", "train"))


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
        if not count:
            raise ValueError(f"No data for {path}")
    temporary.replace(path)
    print(f"{path}: {count:,} rows", flush=True)


def flatten(value):
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def prepare_summary(name, open_split, read_rows, article_key, summary_key):
    # Keep test and validation intact where possible. Drop any repeating training input.
    seen_ids, seen_texts = set(), set()
    for split, source_split in SPLITS:
        discarded = {"empty": 0, "duplicate": 0}

        def rows():
            with open_split(source_split) as stream:
                for row in read_rows(stream):
                    key = str(row["id"])
                    text, summary = flatten(row[article_key]), flatten(row[summary_key])
                    if not text or not summary:
                        discarded["empty"] += 1
                        continue
                    digest = sha256(text.casefold().encode("utf-8")).digest()
                    if key in seen_ids or digest in seen_texts:
                        discarded["duplicate"] += 1
                        continue
                    seen_ids.add(key)
                    seen_texts.add(digest)
                    yield key, text, summary

        save_csv(PREPARED / name / f"{split}.csv", ("id", "text", "summary"), rows())
        print(f"  filtered: {discarded}", flush=True)


def prepare_indosum():
    with tarfile.open(RAW / "indosum/indosum.tar.gz") as archive:
        prepare_summary(
            "indosum",
            lambda split: archive.extractfile(f"indosum/{split}.01.jsonl"),
            lambda stream: (json.loads(line) for line in stream),
            "paragraphs",
            "summary",
        )


def prepare_fleurs():
    source = pq.ParquetFile(RAW / "fleurs/test.parquet")
    output = PREPARED / "fleurs"
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def rows():
        names = set()
        columns = ["id", "audio", "transcription", "raw_transcription", "num_samples"]
        for batch in source.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                audio = row["audio"]
                name = Path(audio["path"]).name
                waveform = audio["bytes"]
                if name in names or not waveform.startswith(b"RIFF") or waveform[8:12] != b"WAVE":
                    raise ValueError(f"Unexpected FLEURS audio file: {name}")
                if not row["raw_transcription"].strip() or row["num_samples"] <= 0:
                    raise ValueError(f"Invalid FLEURS reference: {name}")
                names.add(name)
                (audio_dir / name).write_bytes(waveform)
                yield name, row["id"], f"audio/{name}", row["raw_transcription"], row["transcription"], row["num_samples"] / 16000

    save_csv(output / "test.csv", ("id", "sentence_id", "audio_path", "transcript", "normalized_transcript", "duration_seconds"), rows())


def prepare_lumina():
    source = RAW / "lumina"
    output = PREPARED / "lumina"
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    workbook = load_workbook(source / "sentences.xlsx", read_only=True, data_only=True)
    try:
        values = workbook.active.values
        if next(values) != ("Sentence Number", "Text"):
            raise ValueError("Unexpected LUMINA worksheet headers")
        sentences = {int(index): text for index, text in values}
    finally:
        workbook.close()

    def rows():
        selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
        for row in selection:
            clip_id, sentence_id = row["clip_id"], row["sentence_id"]
            target = clips / f"{clip_id}.mp4"
            temporary = target.with_suffix(".part.mp4")
            # The source MP4 can be silent. Keep the full source audio and hold the last frame if needed.
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-y",
                "-i", str(source / f"{clip_id}.mp4"), "-i", str(source / f"{clip_id}.wav"),
                "-map", "0:v:0", "-map", "1:a:0", "-vf", "tpad=stop_mode=clone:stop=-1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(temporary),
            ], check=True)
            temporary.replace(target)
            yield clip_id, sentence_id, clip_id.partition("_")[0], f"clips/{target.name}", sentences[sentence_id]

    save_csv(output / "test.csv", ("id", "sentence_id", "speaker_id", "video_path", "transcript"), rows())


def main():
    tasks = {"indosum": prepare_indosum, "fleurs": prepare_fleurs, "lumina": prepare_lumina}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["all", *tasks], default="all", nargs="?")
    selected = parser.parse_args().dataset
    for name, prepare in tasks.items():
        if selected in ("all", name):
            prepare()


if __name__ == "__main__":
    main()
