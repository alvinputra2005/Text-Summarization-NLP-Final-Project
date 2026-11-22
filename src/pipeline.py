from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from src.gemini_preprocessing import GeminiError, gemini_config, prepare_gemini_sources
from src.media import MediaError, prepare_video
from src.overview import Overview, prepare_evidence, summarize_overview
from src.segmentation import Segment
from src.step_sources import StepSource, prepare_intro, prepare_step_sources
from src.transcript_quality import coding_transcript_status, is_readable_summary, readable_coverage
from src.transcription import DEFAULT_WHISPER_BACKEND, WHISPER_MODEL, transcribe, whisper_backend, whisper_beam_size, whisper_model
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
    model_input: str = ""


@dataclass(frozen=True)
class ProcessingTimes:
    media: float
    transcription: float
    summarization: float


@dataclass(frozen=True)
class YouTubeInspection:
    video: str
    captions: tuple[list[Segment], str] | None
    elapsed_seconds: float


def inspect_youtube(link: str) -> YouTubeInspection:
    started = perf_counter()
    video = youtube_watch_url(link)
    with TemporaryDirectory(prefix="tutorial-subtitles-") as scratch:
        captions = fetch_youtube_captions(video, Path(scratch))
    return YouTubeInspection(video, captions, perf_counter() - started)


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
    transcript_note: str = ""
    status: str = "ready"
    transcription_beam_size: int = 1
    preprocessing: str = "Lokal"
    preprocessing_note: str = ""

    @property
    def summary_points(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.summary for step in self.steps if step.summary))

    def find_steps(self, query: str) -> list[tuple[int, Step]]:
        terms = query.casefold().split()
        matches = []
        for index, step in enumerate(self.steps):
            text = f"{step.summary} {step.transcript}".casefold()
            if all(term in text for term in terms):
                matches.append((index, step))
        return matches


def summarize_segments(
    segments: list[Segment], progress: Progress, *, sources: list[StepSource] | None = None,
) -> tuple[list[Step], str, Overview]:
    if sources is None:
        progress(0.58, "Mengelompokkan transkrip per langkah…")
        sources = prepare_step_sources(segments)
    if not sources:
        raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")

    progress(0.64, "Memuat mBART-50…")
    from src.summarization import model_reference, summarize

    reference = model_reference()
    tokenizer, model = load_cached_mbart(reference) if any(source.text for source in sources) else (None, None)
    steps = []
    for index, source in enumerate(sources):
        progress(0.66 + 0.27 * index / len(sources), f"Meringkas langkah {index + 1} dari {len(sources)}…")
        segment = source.segment
        text, excerpt = "", False
        if source.text:
            result = summarize(source.text, tokenizer, model, required_clause=source.required_clause)
            text, excerpt = result.text, result.is_excerpt
            if not is_readable_summary(text):
                text = source.text if is_readable_summary(source.text) else ""
                excerpt = bool(text)
        steps.append(Step(segment.start, segment.end, segment.transcript, text, excerpt, source.text))
    progress(0.95, "Membuat pengantar video…")
    evidence = tuple(step.model_input for step in steps if step.summary)
    context = prepare_intro(evidence)
    intro = Overview("", evidence, False)
    if context:
        result = summarize(context, tokenizer, model, required_clause=context.removeprefix("Video ini membahas "))
        intro = Overview(result.text, evidence, result.is_excerpt)
    return steps, reference, intro


def finish_guide(
    video: bytes | str, segments: list[Segment], started: float, transcribed_in: float, progress: Progress,
    *, overview: bool, transcript_source: str | None = None, transcript_note: str = "",
    transcription_beam_size: int | None = None,
) -> Guide:
    summary_started = perf_counter()
    overview_result = None
    preprocessing = "Lokal"
    preprocessing_note = ""
    status = coding_transcript_status(segments)
    if status != "ready":
        preprocessing = "Dilewati"
        from src.summarization import model_reference

        steps, reference = [], model_reference()
    elif overview:
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
        sources = []
        try:
            config = gemini_config()
            if config:
                progress(0.55, "Menyeleksi inti transkrip dengan Gemini…")
                sources = prepare_gemini_sources(segments, config)
                if sources:
                    preprocessing = f"Gemini · {config.model}"
                else:
                    preprocessing_note = "Belum ada cuplikan Gemini yang lolos pemeriksaan. Aplikasi memakai preprocessing lokal."
        except GeminiError as error:
            preprocessing_note = f"{error} Aplikasi memakai preprocessing lokal."
        if sources:
            steps, reference, overview_result = summarize_segments(segments, progress, sources=sources)
        else:
            steps, reference, overview_result = summarize_segments(segments, progress)
        if not any(step.summary for step in steps):
            status = "unclear"
    finished = perf_counter()
    guide = Guide(
        video, steps, reference, processing_seconds=finished - started, transcription_backend=whisper_backend(),
        transcription_model=whisper_model(), status=status,
        transcription_beam_size=transcription_beam_size or whisper_beam_size(),
        timings=ProcessingTimes(
            media=max(0, summary_started - started - transcribed_in),
            transcription=transcribed_in,
            summarization=finished - summary_started,
        ),
        transcript=tuple(segments), overview=overview_result, transcript_source=transcript_source,
        transcript_note=transcript_note, preprocessing=preprocessing, preprocessing_note=preprocessing_note,
    )
    progress(1.0, "Panduan siap." if status == "ready" else "Media selesai diperiksa.")
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


def build_youtube_guide(
    link: str | YouTubeInspection, progress: Progress, *, overview: bool = False, prefer_captions: bool = False,
) -> Guide:
    checked = link if isinstance(link, YouTubeInspection) else None
    inspection_time = checked.elapsed_seconds if checked else 0.0
    started = perf_counter() - inspection_time
    progress(0.05, "Memeriksa link YouTube…")
    video = youtube_watch_url(checked.video if checked else link)
    transcript_source = None
    transcript_note = ""
    corrected_beam = None
    with TemporaryDirectory(prefix="tutorial-youtube-") as scratch:
        transcription_started = perf_counter()
        captions = checked.captions if checked else None
        if checked is None and (overview or prefer_captions):
            progress(0.12, "Memeriksa subtitle bahasa Indonesia…")
            captions = fetch_youtube_captions(video, Path(scratch))
        if captions is not None:
            segments, transcript_source = captions
            caption_coverage = readable_coverage(segments)
            if len(segments) >= 12 and caption_coverage < 0.25:
                progress(0.18, "Subtitle sulit dibaca. Memeriksa ulang audio dengan Whisper…")
                try:
                    audio = download_youtube_audio(video, Path(scratch))
                    revised = transcribe(audio, retry_noisy=True)
                except (MediaError, OSError, RuntimeError, ValueError):
                    transcript_note = "Subtitle sulit dibaca dan transkripsi ulang gagal. Periksa segmen sumber."
                else:
                    caption_evidence = sum(bool(step.required_clause) for step in prepare_step_sources(segments))
                    revised_evidence = sum(bool(step.required_clause) for step in prepare_step_sources(revised))
                    if readable_coverage(revised) > caption_coverage and revised_evidence >= caption_evidence:
                        segments, transcript_source = revised, None
                        corrected_beam = max(3, whisper_beam_size())
                        transcript_note = "Audio ditranskripsi ulang dengan Whisper karena subtitle kurang terbaca."
                    else:
                        transcript_note = "Subtitle dipakai setelah membandingkan keterbacaan dan contoh teknis dengan hasil Whisper."
        else:
            progress(0.18, "Mengambil audio YouTube…")
            audio = download_youtube_audio(video, Path(scratch))
            progress(0.36, "Mentranskripsi narasi dengan Whisper…")
            transcription_started = perf_counter()
            segments = transcribe(audio)
        transcribed_in = perf_counter() - transcription_started + inspection_time
    return finish_guide(
        video, segments, started, transcribed_in, progress, overview=overview, transcript_source=transcript_source,
        transcript_note=transcript_note, transcription_beam_size=corrected_beam,
    )
