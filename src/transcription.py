from pathlib import Path
from threading import Lock

import streamlit as st

from src.media import MediaError
from src.segmentation import Segment


WHISPER_MODEL = "base"
whisper_lock = Lock()


@st.cache_resource(show_spinner=False)
def load_whisper():
    from whisper import load_model

    return load_model(WHISPER_MODEL, device="cpu")


def transcribe(media: Path) -> list[Segment]:
    from whisper import load_audio

    try:
        audio = load_audio(str(media))
    except RuntimeError as error:
        raise MediaError("Audio gagal dibaca. Coba sumber video lain dengan suara yang bisa diputar.") from error
    if not audio.any():
        raise MediaError("Audio hening. Pilih video dengan narasi yang terdengar.")
    with whisper_lock:
        result = load_whisper().transcribe(audio, language="id", task="transcribe", fp16=False)
    return [Segment(item["start"], item["end"], item["text"]) for item in result["segments"]]
