"""Prepare a human-speech sample from an official NSS-ID Imperative split."""

import argparse
import csv
from pathlib import Path
import re
import tarfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nss-id"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/nss-id"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--per-speaker", type=int, default=8)
    args = parser.parse_args()
    if args.per_speaker < 1:
        parser.error("Gunakan minimal satu klip per penutur.")
    metadata = pd.read_parquet(args.raw_dir / f"{args.split}.parquet")
    candidates = metadata.loc[
        (metadata["category"] == "Imperative") & (metadata["split"] == args.split)
        & (metadata["speaker_type"] == "human") & ~metadata["is_synthetic"],
        ["audio_path", "speaker_id", "sentence_id", "transcript"],
    ].sort_values("audio_path")
    rows = []
    for _, group in candidates.groupby("speaker_id", sort=True):
        rows.extend(group.sample(frac=1, random_state=42).drop_duplicates("sentence_id").head(args.per_speaker).to_dict("records"))
    if not rows:
        raise ValueError(f"Tidak ada klip manusia pada split {args.split} Imperative.")
    audio = args.output_dir / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    wanted = {row["audio_path"]: index for index, row in enumerate(rows)}
    written = set()
    with tarfile.open(args.raw_dir / "Imperative.tar") as archive:
        for member in archive:
            index = wanted.get(member.name)
            if index is None or not member.isfile():
                continue
            speaker = rows[index]["speaker_id"]
            if not re.fullmatch(r"[FM]\d+", speaker):
                raise ValueError("Format ID penutur dataset tidak valid.")
            identifier = f"nss-id-{speaker}-{index + 1:03}"
            with archive.extractfile(member) as source:
                (audio / f"{identifier}.wav").write_bytes(source.read())
            rows[index]["id"] = identifier
            rows[index]["source_audio"] = member.name
            rows[index]["audio_path"] = f"audio/{identifier}.wav"
            written.add(index)
            if len(written) == len(rows):
                break
    if len(written) != len(rows):
        raise ValueError("Sebagian file tidak ditemukan pada arsip audio NSS-ID.")
    with (args.output_dir / f"{args.split}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("id", "audio_path", "speaker_id", "sentence_id", "transcript", "source_audio"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"NSS-ID Imperative: {len(rows)} klip dari {candidates['speaker_id'].nunique()} penutur pada split {args.split}.")


if __name__ == "__main__":
    main()
