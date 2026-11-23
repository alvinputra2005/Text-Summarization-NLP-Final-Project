"""Create speaker-disjoint manifest views of the prepared LUMINA clips."""

import argparse
import csv
import os
from pathlib import Path


def split_lumina(source: Path, destination: Path) -> dict[str, int]:
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"id", "speaker_id", "video_path", "transcript"}.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest LUMINA harus berisi id, speaker_id, video_path, dan transcript.")
        fields = reader.fieldnames
        rows = list(reader)
    speakers = sorted({row["speaker_id"] for row in rows})
    if len(speakers) < 4 or "" in speakers or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Perlu minimal empat penutur LUMINA dengan ID klip unik.")
    if source.parent.resolve() == destination.resolve():
        raise ValueError("Simpan split baru di folder terpisah dari manifest masukan.")

    held_out = max(1, len(speakers) // 4)
    groups = {
        "train": set(speakers[:-2 * held_out]),
        "validation": set(speakers[-2 * held_out:-held_out]),
        "test": set(speakers[-held_out:]),
    }
    for row in rows:
        media = (source.parent / row["video_path"]).resolve()
        if not media.is_file() or not row["transcript"].strip():
            raise ValueError(f"Media atau transkrip LUMINA tidak tersedia untuk {row['id']}.")
        row["video_path"] = Path(os.path.relpath(media, destination.resolve())).as_posix()
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, group in groups.items():
        selected = [row for row in rows if row["speaker_id"] in group]
        with (destination / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        counts[name] = len(selected)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/lumina/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/lumina_split"))
    args = parser.parse_args()
    print(split_lumina(args.source, args.output_dir))


if __name__ == "__main__":
    main()
