from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from src.media import MediaError, prepare_video
from src.overview import Overview, prepare_evidence, summarize_overview
from src.segmentation import Segment, group_segments
from src.transcription import DEFAULT_WHISPER_BACKEND, WHISPER_MODEL, transcribe, whisper_backend
from src.youtube import download_youtube_audio, fetch_youtube_captions, youtube_watch_url


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
class ProcessingTimes:
    media: float
    transcription: float
    summarization: float


@dataclass(frozen=True)
class Guide:
    video: bytes | str
    steps: list[Step]
    model: str
    processing_seconds: float = 0.0
    transcription_model: str = WHISPER_MODEL
    transcription_backend: str = DEFAULT_WHISPER_BACKEND
    timings: ProcessingTimes | None = None
    transcript: tuple[Segment, ...] = ()
    overview: Overview | None = None
    transcript_source: str | None = None

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


def finish_guide(
    video: bytes | str, segments: list[Segment], started: float, transcribed_in: float, progress: Progress,
    *, overview: bool, transcript_source: str | None = None,
) -> Guide:
    summary_started = perf_counter()
    overview_result = None
    if overview:
        from src.summarization import model_reference

        progress(0.58, "Menyiapkan teks dari seluruh video…")
        evidence = prepare_evidence(segments)
        if not evidence:
            raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")
        reference = model_reference()
        progress(0.66, "Membuat ringkasan keseluruhan dengan mBART-50…")
        tokenizer, model = load_cached_mbart(reference)
        overview_result = summarize_overview(evidence, tokenizer, model)
        steps = []
    else:
        steps, reference = summarize_segments(segments, progress)
    finished = perf_counter()
    guide = Guide(
        video, steps, reference, processing_seconds=finished - started, transcription_backend=whisper_backend(),
        timings=ProcessingTimes(
            media=max(0, summary_started - started - transcribed_in),
            transcription=transcribed_in,
            summarization=finished - summary_started,
        ),
        transcript=tuple(segments), overview=overview_result, transcript_source=transcript_source,
    )
    progress(1.0, "Panduan siap.")
    return guide


def build_guide(video: bytes, progress: Progress, *, overview: bool = False) -> Guide:
    if not video:
        raise MediaError("Berkas MP4 kosong. Pilih video lain.")
    started = perf_counter()
    with TemporaryDirectory(prefix="tutorial-summarizer-") as scratch:
        upload = Path(scratch) / "upload.mp4"
        upload.write_bytes(video)
        progress(0.05, "Memeriksa format, audio, dan kompatibilitas video…")
        playable = prepare_video(upload)

        progress(0.22, "Mentranskripsi narasi dengan Whisper…")
        transcription_started = perf_counter()
        segments = transcribe(playable)
        transcribed_in = perf_counter() - transcription_started
        playback = video if playable == upload else playable.read_bytes()
    return finish_guide(playback, segments, started, transcribed_in, progress, overview=overview)


def build_youtube_guide(link: str, progress: Progress, *, overview: bool = False) -> Guide:
    started = perf_counter()
    progress(0.05, "Memeriksa link YouTube…")
    video = youtube_watch_url(link)
    transcript_source = None
    with TemporaryDirectory(prefix="tutorial-youtube-") as scratch:
        transcription_started = perf_counter()
        captions = None
        if overview:
            progress(0.12, "Memeriksa subtitle bahasa Indonesia…")
            captions = fetch_youtube_captions(video, Path(scratch))
        if captions is not None:
            segments, transcript_source = captions
        else:
            progress(0.18, "Mengambil audio YouTube…")
            audio = download_youtube_audio(video, Path(scratch))
            progress(0.36, "Mentranskripsi narasi dengan Whisper…")
            transcription_started = perf_counter()
            segments = transcribe(audio)
        transcribed_in = perf_counter() - transcription_started
    return finish_guide(
        video, segments, started, transcribed_in, progress, overview=overview, transcript_source=transcript_source,
    )
