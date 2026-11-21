"""Download a small, reproducible set of paired audio and video clips from LUMINA v4."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import subprocess
from urllib.parse import urlencode


API = "https://data.mendeley.com/public-api/datasets/8fw93k4rny"
DESTINATION = Path("data/raw/lumina")
CLIP = re.compile(r"P(\d+)_S(\d+)", re.IGNORECASE)
PER_SPEAKER = 12


def fetch(url: str) -> bytes:
    return subprocess.check_output(["curl", "-fLsS", "--retry", "2", "--max-time", "30", url])


def list_files(folder: str, pages: int = 1) -> dict[str, dict]:
    files = {}
    for page in range(pages):
        params = urlencode({"folder_id": folder, "version": 4, "$start": page * 1000, "$limit": 1000})
        batch = json.loads(fetch(f"{API}/files?{params}"))
        files.update((Path(item["filename"]).stem.upper(), item) for item in batch)
        if len(batch) < 1000:
            break
    return files


def download(item: dict, destination: Path) -> None:
    details = item["content_details"]
    digest = details["sha256_hash"]
    if destination.is_file() and sha256(destination.read_bytes()).hexdigest() == digest:
        return
    payload = fetch(details["download_url"])
    if sha256(payload).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch: {destination.name}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    folders = json.loads(fetch(f"{API}/folders/4"))
    rng = random.Random(42)
    selected = []
    for parent in (item for item in folders if "parent_id" not in item):
        children = {item["name"]: item["id"] for item in folders if item.get("parent_id") == parent["id"]}
        audio = list_files(children["audio"], pages=2)
        video = list_files(children["video"], pages=2)
        pairs = defaultdict(list)
        for key in sorted(audio.keys() & video.keys()):
            match = CLIP.fullmatch(key)
            if match:
                pairs[match[1]].append(key)
        speakers = [speaker for speaker in sorted(pairs) if len(pairs[speaker]) >= PER_SPEAKER][:2]
        if len(speakers) != 2:
            raise ValueError(f"Expected two speakers with paired clips in {parent['name']}")
        for speaker in speakers:
            for key in sorted(rng.sample(pairs[speaker], PER_SPEAKER)):
                selected.append((key, audio[key], video[key]))

    worksheet, = list_files("root").values()
    jobs = [(worksheet, DESTINATION / "sentences.xlsx")]
    manifest = []
    for key, audio, video in selected:
        match = CLIP.fullmatch(key)
        jobs.extend([(audio, DESTINATION / f"{key}.wav"), (video, DESTINATION / f"{key}.mp4")])
        manifest.append({"clip_id": key, "sentence_id": int(match[2]), "audio_id": audio["id"], "video_id": video["id"]})
    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(lambda job: download(*job), jobs))
    (DESTINATION / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"LUMINA: {len(manifest)} MP4/WAV pairs and the source sentence workbook downloaded.")


if __name__ == "__main__":
    main()
