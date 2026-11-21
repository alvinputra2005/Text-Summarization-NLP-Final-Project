import json
from pathlib import Path
import subprocess


class MediaError(ValueError):
    pass


def run_tool(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as error:
        raise MediaError(f"{args[0]} tidak ditemukan. Pasang FFmpeg dan buka ulang PowerShell.") from error
    if result.returncode:
        raise MediaError("Video tidak bisa dibaca atau audio gagal didekode. Coba ekspor ulang MP4.")
    return result.stdout


def prepare_video(upload: Path) -> Path:
    metadata = json.loads(
        run_tool(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(upload)])
    )
    container = metadata.get("format", {})
    formats = container.get("format_name", "").split(",")
    brand = container.get("tags", {}).get("major_brand", "").strip()
    if "mp4" not in formats or brand == "qt":
        raise MediaError("Berkas harus memakai format MP4.")

    streams = metadata.get("streams", [])
    videos = [s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in streams if s["codec_type"] == "audio"]
    if not videos or not audios:
        raise MediaError("MP4 harus memiliki jalur video dan audio. Pilih video yang berisi narasi.")
    try:
        duration = float(container["duration"])
    except (KeyError, ValueError):
        duration = 0
    if not 0 < duration < float("inf"):
        raise MediaError("Durasi video tidak bisa dibaca. Coba ekspor ulang MP4.")

    video, audio = videos[0], audios[0]
    browser_ready = (
        len(streams) == 2
        and video.get("codec_name") == "h264"
        and video.get("pix_fmt") == "yuv420p"
        and audio.get("codec_name") == "aac"
        and video.get("start_time") == audio.get("start_time") == "0.000000"
    )
    if browser_ready:
        return upload

    playable = upload.with_name("playback.mp4")
    run_tool([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(upload),
        "-map", f"0:{video['index']}", "-map", f"0:{audio['index']}",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-af", "aresample=async=1:first_pts=0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-c:a", "aac", "-movflags", "+faststart", str(playable),
    ])
    return playable
