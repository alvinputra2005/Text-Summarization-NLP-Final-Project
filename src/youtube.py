import logging
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

from src.media import MediaError


log = logging.getLogger(__name__)
YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com",
}
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def youtube_watch_url(link: str) -> str:
    link = link.strip()
    if "://" not in link:
        link = f"https://{link}"
    try:
        url = urlsplit(link)
    except ValueError as error:
        raise MediaError("Link YouTube tidak valid.") from error

    host = url.netloc.lower()
    path = url.path.rstrip("/")
    parts = path.split("/")
    video_id = ""
    if url.scheme in {"https", "http"}:
        if host == "youtu.be" and len(parts) == 2:
            video_id = parts[1]
        elif host in YOUTUBE_HOSTS:
            if path == "/watch":
                ids = parse_qs(url.query, keep_blank_values=True).get("v", [])
                if len(ids) == 1:
                    video_id = ids[0]
            elif len(parts) == 3 and parts[1] in {"shorts", "live", "embed"}:
                video_id = parts[2]
    if not VIDEO_ID.fullmatch(video_id):
        raise MediaError("Masukkan link ke satu video YouTube, misalnya youtube.com/watch?v=… atau youtu.be/…")
    return f"https://www.youtube.com/watch?v={video_id}"


def download_youtube_audio(video: str, directory: Path) -> Path:
    """Download one validated YouTube video source into the caller's temporary directory."""
    command = [
        sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
        "--js-runtimes", "node", "--no-progress", "--encoding", "utf-8",
        "--socket-timeout", "15", "--retries", "2", "--match-filter", "!is_live",
        "--format", "bestaudio/best", "--output", "source.%(ext)s",
        "--print", "after_move:filepath", "--no-simulate", "--", video,
    ]
    try:
        result = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise MediaError("Pengambilan audio YouTube terlalu lama. Coba lagi atau pilih video lebih pendek.") from error

    if result.returncode:
        log.warning("Pengambilan audio YouTube gagal: %s", result.stderr.strip()[-1000:])
        if "No module named yt_dlp" in result.stderr:
            raise MediaError("yt-dlp belum terpasang. Pasang dependensi dari requirements.txt.")
        raise MediaError(
            "Audio YouTube tidak bisa diambil. Coba video publik lain atau gunakan tab Unggah MP4."
        )

    filenames = result.stdout.strip().splitlines()
    if len(filenames) == 1:
        audio = (directory / filenames[0]).resolve()
        if audio.parent == directory.resolve() and audio.is_file() and audio.stat().st_size > 0:
            return audio
    raise MediaError("Audio belum tersedia. Pilih video YouTube yang sudah selesai, seperti rekaman tutorial.")
