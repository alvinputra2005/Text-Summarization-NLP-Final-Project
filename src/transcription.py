from pathlib import Path
from threading import Lock

import streamlit as st
import whisper

from src.media import MediaError
from src.segmentation import Segment


WHISPER_MODEL = "base"
whisper_lock = Lock()


@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model(WHISPER_MODEL, device="cpu")


def transcribe(video: Path) -> list[Segment]:
    try:
        audio = whisper.load_audio(str(video))
    except RuntimeError as error:
        raise MediaError("Audio gagal didekode. Coba ekspor ulang MP4 dengan audio AAC.") from error
    if not audio.any():
        raise MediaError("Audio hening. Pilih video dengan narasi yang terdengar.")
    with whisper_lock:
        result = load_whisper().transcribe(audio, language="id", task="transcribe", fp16=False)
    return [Segment(item["start"], item["end"], item["text"]) for item in result["segments"]]
