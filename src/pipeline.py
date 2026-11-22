from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from src.media import MediaError, prepare_video
from src.segmentation import Segment, group_segments
from src.transcription import WHISPER_MODEL, transcribe
from src.youtube import download_youtube_audio, youtube_watch_url


Progress = Callable[[float, str], None]


@st.cache_resource(show_spinner=False)
def load_cached_mbart(reference: str):
    from src.summarization import load_mbart

    return load_mbart(reference)


@dataclass(frozen=True)
class Step:
    start: float
    end: float
    transcript: str
    summary: str
    is_excerpt: bool


@dataclass(frozen=True)
class Guide:
    video: bytes | str
    steps: list[Step]
    model: str
    processing_seconds: float = 0.0
    transcription_model: str = WHISPER_MODEL

    def find_steps(self, query: str) -> list[tuple[int, Step]]:
        terms = query.casefold().split()
        matches = []
        for index, step in enumerate(self.steps):
            text = f"{step.summary} {step.transcript}".casefold()
            if all(term in text for term in terms):
                matches.append((index, step))
        return matches


def summarize_segments(segments: list[Segment], progress: Progress) -> tuple[list[Step], str]:
    progress(0.58, "Mengelompokkan transkrip per langkah…")
    groups = group_segments(segments)
    if not groups:
        raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")

    progress(0.64, "Memuat mBART-50…")
    from src.summarization import model_reference, summarize

    reference = model_reference()
    tokenizer, model = load_cached_mbart(reference)
    steps = []
    for index, group in enumerate(groups):
        progress(0.66 + 0.32 * index / len(groups), f"Meringkas langkah {index + 1} dari {len(groups)}…")
        result = summarize(group.text, tokenizer, model)
        steps.append(Step(group.start, group.end, group.transcript, result.text, result.is_excerpt))
    return steps, reference


def build_guide(video: bytes, progress: Progress) -> Guide:
    if not video:
        raise MediaError("Berkas MP4 kosong. Pilih video lain.")
    started = perf_counter()
    with TemporaryDirectory(prefix="tutorial-summarizer-") as scratch:
        upload = Path(scratch) / "upload.mp4"
        upload.write_bytes(video)
        progress(0.05, "Memeriksa format, audio, dan kompatibilitas video…")
        playable = prepare_video(upload)

        progress(0.22, "Mentranskripsi narasi dengan Whisper…")
        segments = transcribe(playable)
        playback = video if playable == upload else playable.read_bytes()
    steps, reference = summarize_segments(segments, progress)
    guide = Guide(playback, steps, reference, processing_seconds=perf_counter() - started)
    progress(1.0, "Panduan siap.")
    return guide


def build_youtube_guide(link: str, progress: Progress) -> Guide:
    started = perf_counter()
    progress(0.05, "Memeriksa link YouTube…")
    video = youtube_watch_url(link)
    with TemporaryDirectory(prefix="tutorial-youtube-") as scratch:
        progress(0.12, "Mengambil audio YouTube…")
        audio = download_youtube_audio(video, Path(scratch))
        progress(0.22, "Mentranskripsi narasi dengan Whisper…")
        segments = transcribe(audio)
    steps, reference = summarize_segments(segments, progress)
    guide = Guide(video, steps, reference, processing_seconds=perf_counter() - started)
    progress(1.0, "Panduan siap.")
    return guide
