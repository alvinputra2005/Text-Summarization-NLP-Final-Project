"""Build video-disjoint audio and subtitle splits for the Whisper pilot."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_video_experiment import captions, clip_text
from src.media import run_tool
from src.youtube import download_youtube_audio


DEFAULT_VIDEOS = {
    "train": ("Ar1xxIsyuvI", "JUE9vPXefGI"),
    "validation": ("Z4qfMhx4XzQ",),
    "test": ("I8mpr4AvZiQ",),
}


def training_windows(cues, limit):
    starts = [float(cue[0]) for cue in cues]
    last_end = max(float(cue[1]) for cue in cues)
    windows = []
    for start in range(30, int(last_end) - 11, 12):
        end = start + 12
        text = clip_text(cues, start, end)
        if len(text.split()) >= 10 and any(start <= t < end for t in starts):
            windows.append((start, end, text))
    if len(windows) > limit:
        return [windows[min(len(windows) - 1, int((i + 0.5) * len(windows) / limit))] for i in range(limit)]
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=DEFAULT_VIDEOS, default="train")
    parser.add_argument("--video-ids", nargs="+")
    parser.add_argument("--clips-per-video", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/whisper_video"))
    args = parser.parse_args()
    video_ids = args.video_ids or DEFAULT_VIDEOS[args.split]
    if len(set(video_ids)) != len(video_ids) or args.clips_per_video < 1:
        parser.error("ID video harus unik dan jumlah klip minimal satu.")

    raw = Path("data/raw/coding_video")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / f"{args.split}.csv"
    if manifest.exists():
        parser.error("Manifest sudah ada. Gunakan folder keluaran lain untuk membuat split baru.")
    for split, reserved in DEFAULT_VIDEOS.items():
        if split == args.split:
            continue
        existing = folder / f"{split}.csv"
        if existing.is_file():
            with existing.open(encoding="utf-8", newline="") as file:
                reserved = {row["video_id"] for row in csv.DictReader(file)}
        if set(video_ids).intersection(reserved):
            parser.error(f"Video pilihan sudah dipakai atau dicadangkan untuk split {split}.")
    (folder / "audio").mkdir(exist_ok=True)
    rows = []
    for video in video_ids:
        cues = captions(raw, video, "id")
        source = json.loads((raw / f"{video}.json").read_text(encoding="utf-8"))["source"]
        windows = training_windows(cues, args.clips_per_video)
        if not windows:
            raise ValueError(f"Tidak ada subtitle yang cukup panjang untuk video {video}.")
        with TemporaryDirectory(dir=folder) as scratch:
            media = download_youtube_audio(f"https://www.youtube.com/watch?v={video}", Path(scratch))
            for start, end, text in windows:
                identifier = f"{video}-{start:04d}"
                relative_audio = f"audio/{identifier}.wav"
                run_tool([
                    "ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", str(start), "-i", str(media),
                    "-t", str(end - start), "-vn", "-ac", "1", "-ar", "16000", str(folder / relative_audio),
                ])
                rows.append({
                    "id": identifier, "video_id": video, "start": start, "end": end, "audio_path": relative_audio,
                    "transcript": text, "transcript_source": source,
                })
    with manifest.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} klip dari {len(video_ids)} video disimpan di {manifest}.")


if __name__ == "__main__":
    main()
