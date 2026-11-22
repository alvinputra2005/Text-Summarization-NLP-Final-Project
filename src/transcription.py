import os
from pathlib import Path
from threading import Lock

import streamlit as st

from src.media import MediaError
from src.segmentation import Segment


WHISPER_MODEL = "base"
DEFAULT_WHISPER_BACKEND = "faster"
PYTHON_PROMPT = (
    "Tutorial pemrograman Python dalam bahasa Indonesia. Istilah teknis: variabel, tipe data, string, "
    "integer, float, boolean, True, False, not, and, or, XOR, input(), print(), pandas, CSV."
)
whisper_lock = Lock()


def whisper_model() -> str:
    return os.environ.get("WHISPER_MODEL_ID", "").strip() or WHISPER_MODEL


def whisper_beam_size() -> int:
    value = os.environ.get("WHISPER_BEAM_SIZE", "").strip() or "1"
    if value not in {"1", "2", "3", "4", "5"}:
        raise MediaError("WHISPER_BEAM_SIZE harus berupa angka 1 sampai 5.")
    return int(value)


def whisper_backend() -> str:
    backend = os.environ.get("WHISPER_BACKEND", "").strip().lower() or DEFAULT_WHISPER_BACKEND
    if backend not in {"faster", "openai"}:
        raise MediaError("WHISPER_BACKEND harus bernilai faster atau openai.")
    return backend


@st.cache_resource(show_spinner=False)
def load_whisper(reference: str = WHISPER_MODEL):
    from whisper import load_model

    return load_model(reference, device="cpu")


@st.cache_resource(show_spinner=False)
def load_faster_whisper(reference: str = WHISPER_MODEL):
    from faster_whisper import WhisperModel

    return WhisperModel(
        reference, device="cpu", compute_type="int8", cpu_threads=min(4, os.cpu_count() or 1),
    )


def transcribe(media: Path, *, retry_noisy: bool = False) -> list[Segment]:
    beam_size = whisper_beam_size()
    decoding = {"beam_size": max(3, beam_size), "initial_prompt": PYTHON_PROMPT} if retry_noisy else {"beam_size": beam_size}
    reference = whisper_model()
    if whisper_backend() == "faster":
        from av.error import FFmpegError
        from faster_whisper.audio import decode_audio

        try:
            audio = decode_audio(str(media))
        except FFmpegError as error:
            raise MediaError("Audio gagal dibaca. Coba sumber video lain dengan suara yang bisa diputar.") from error
        if not audio.any():
            return []
        with whisper_lock:
            segments, _ = load_faster_whisper(reference).transcribe(
                audio, language="id", task="transcribe", best_of=1,
                vad_filter=True, word_timestamps=True, **decoding,
            )
            return [Segment(float(item.start), float(item.end), item.text) for item in segments]

    from whisper import load_audio

    try:
        audio = load_audio(str(media))
    except RuntimeError as error:
        raise MediaError("Audio gagal dibaca. Coba sumber video lain dengan suara yang bisa diputar.") from error
    if not audio.any():
        return []
    with whisper_lock:
        result = load_whisper(reference).transcribe(audio, language="id", task="transcribe", fp16=False, **decoding)
    return [Segment(item["start"], item["end"], item["text"]) for item in result["segments"]]
