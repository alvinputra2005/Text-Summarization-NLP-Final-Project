from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from subprocess import SubprocessError
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from src.gemini_preprocessing import GEMINI_QUOTA_NOTICE, GeminiError, GeminiQuotaError, gemini_config, prepare_gemini_sources
from src.media import MediaError, prepare_video
from src.overview import Overview, prepare_evidence, summarize_overview
from src.segmentation import Segment
from src.step_sources import StepSource, prepare_intro, prepare_step_sources
from src.transcript_cleaning import clean_utterance
from src.transcript_quality import coding_transcript_status, is_readable_summary, readable_coverage
from src.transcription import DEFAULT_WHISPER_BACKEND, WHISPER_MODEL, transcribe, whisper_backend, whisper_beam_size, whisper_model
from src.youtube import download_youtube_audio, fetch_youtube_captions, youtube_watch_url
from src.string_sources import is_string_lesson, order_string_points, string_speech_points
from src.video_outline import (
    add_distinct_sources, explicit_code_sources, is_lesson_chatter, limit_tutorial_sources, prepare_local_outline,
    recover_tutorial_sources,
)


Progress = Callable[[float, str], None]
log = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_cached_mbart(reference: str):
    from src.summarization import load_mbart

    return load_mbart(reference)


def load_guide_model(reference: str):
    from src.summarization import PUBLIC_MODEL

    try:
        tokenizer, model = load_cached_mbart(reference)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if Path(reference).suffix.lower() != ".json":
            raise
        log.warning("Checkpoint eksperimen tidak dapat dimuat (%s). Memakai mBART bawaan.", type(error).__name__)
        tokenizer, model = load_cached_mbart(PUBLIC_MODEL)
        reference = f"{PUBLIC_MODEL} · checkpoint eksperimen tidak tersedia"
    return tokenizer, model, model.identity if tokenizer is None else reference


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
class VisualProcessing:
    frames: int
    steps: int
    seconds: float
    note: str = ""
    gemini_model: str = ""
    gemini_evidence: int = 0
    gemini_seconds: float = 0.0
    gemini_context_changes: int = 0
    gemini_code_examples: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class YouTubeInspection:
    video: str
    captions: tuple[list[Segment], str] | None
    elapsed_seconds: float

    @property
    def captions_need_review(self) -> bool:
        return self.captions is not None and captions_need_review(self.captions[0])


def captions_need_review(segments: list[Segment]) -> bool:
    return len(segments) >= 12 and readable_coverage(segments) < 0.25


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
    visual: VisualProcessing | None = None
    gemini_quota_exhausted: bool = False
    model_setting: str = ""

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


def select_guide_sources(segments: list[Segment], sources: list[StepSource] | None = None) -> list[StepSource]:
    string_lesson = is_string_lesson(segments)
    if string_lesson:
        verified = [source for source in sources or () if source.required_clause]
        curated = verified or order_string_points(string_speech_points(segments))
        sources = add_distinct_sources(curated, explicit_code_sources(segments)) or [StepSource(segments[0], "")]
    elif sources is None:
        sources = prepare_local_outline(segments, recover_tutorial_sources(segments))
    else:
        sources = add_distinct_sources(sources, recover_tutorial_sources(segments))
    selected = []
    for source in limit_tutorial_sources(sources):
        cleaned, _ = clean_utterance(source.text)
        if source.required_clause and source.required_clause not in cleaned:
            cleaned = source.text
        selected.append(replace(source, text=cleaned))
    return selected


def summarize_segments(
    segments: list[Segment], progress: Progress, *, sources: list[StepSource] | None = None, checkpoint: str = "",
) -> tuple[list[Step], str, Overview]:
    if sources is None:
        progress(0.58, "Mengelompokkan transkrip per langkah…")
    sources = select_guide_sources(segments, sources)
    if not sources:
        raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")

    from src.summarization import model_reference, summarize

    reference = checkpoint or model_reference()
    needs_model = any(source.text and not (source.allow_source_excerpt and source.required_clause == source.text)
                      for source in sources)
    if needs_model:
        progress(0.64, "Memuat model peringkasan…")
        try:
            tokenizer, model, reference = load_guide_model(reference)
        except (OSError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and not any(
                message in str(error).casefold() for message in ("out of memory", "not enough memory", "can't allocate memory")
            ):
                raise
            log.warning("mBART tidak dapat dimuat (%s).", type(error).__name__)
            tokenizer, model = None, None
            needs_model = False
            reference = "Poin dari sumber · mBART tidak dapat dimuat"
    else:
        tokenizer, model = None, None
        if any(source.text for source in sources):
            reference = "Aturan berbasis sumber"
    steps = []
    for index, source in enumerate(sources):
        progress(0.66 + 0.27 * index / len(sources), f"Meringkas langkah {index + 1} dari {len(sources)}…")
        segment = source.segment
        text, excerpt = "", False
        if source.text and source.allow_source_excerpt and source.required_clause == source.text:
            text, excerpt = source.text, True
        elif source.text and (source.required_clause or not is_lesson_chatter(source.text)):
            if needs_model:
                result = summarize(source.text, tokenizer, model, required_clause=source.required_clause,
                                   **({"visual_quote": source.visual_quote} if source.visual_quote else {}))
                text, excerpt = result.text, result.is_excerpt
            if not is_readable_summary(text) or is_lesson_chatter(text):
                text = source.text if source.allow_source_excerpt or is_readable_summary(source.text) else ""
                excerpt = bool(text)
        steps.append(Step(segment.start, segment.end, segment.transcript, text, excerpt, source.text))
    progress(0.95, "Membuat pengantar video…")
    evidence = tuple(step.model_input for step in steps if step.summary)
    context = prepare_intro(evidence)
    intro = Overview("", evidence, False)
    if context and not needs_model:
        intro = Overview(context, evidence, True)
    elif context:
        result = summarize(context, tokenizer, model, required_clause=context.removeprefix("Video ini membahas "))
        intro = Overview(result.text, evidence, result.is_excerpt)
    return steps, reference, intro


def prepare_visual_sources(segments, video, visual_path, youtube_scratch, config, progress, *, screen_only=False):
    if (visual_path is None and youtube_scratch is None) or os.getenv("VIDEO_OCR_ENABLED", "true").lower() in {"0", "off", "false", "no"}:
        return [], None, {}
    try:
        from src.contextual_cleaning import contextual_parts, refine_context
        from src.video_context import (
            contextual_sources, draft_sources, read_with_gemini, scan_context, speech_drafts, visual_branch_sources,
        )
        from src.video_ocr import download_video, tesseract_path

        engine = tesseract_path()
        roi_text = os.getenv("VIDEO_OCR_ROI", "").strip()
        roi = tuple(map(float, roi_text.split())) if roi_text else None
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("Area OCR tidak valid.")
        progress(0.48, "Memeriksa kode yang tampil di layar…")
        media = visual_path if visual_path is not None else download_video(video, youtube_scratch)
        if screen_only:
            context = scan_context(media, segments, engine, roi, None, screen_only=True)
        else:
            context = scan_context(media, segments, engine, roi, config)
    except (ImportError, OSError, ValueError, RuntimeError, SubprocessError) as error:
        log.info("OCR dilewati (%s).", type(error).__name__)
        return [], VisualProcessing(0, 0, 0, "Pembacaan frame tidak tersedia. Panduan memakai transkrip."), {}

    if screen_only:
        sources = visual_branch_sources(segments, context["parts"], [], screen_only=True)
        note = (
            "Langkah disusun dari percabangan yang terbaca konsisten di layar."
            if sources else "Pemeriksaan layar belum menemukan percabangan yang cukup jelas untuk dijadikan langkah."
        )
        return sources, VisualProcessing(context["frame_count"], len(sources), context["ocr_seconds"], note), {}
    if config:
        progress(0.53, "Mencocokkan kode dengan transkrip melalui Gemini…")
    gemini_started = perf_counter()
    vision = read_with_gemini(context["parts"], 2 if config else 0, config=config)
    if config and not vision.get("failed"):
        refinement = refine_context(context["parts"], vision["verified"], config)
        vision["context_changes"] = len(refinement["changes"])
        vision["requests"] = vision.get("requests", 0) + refinement["requests"]
        if refinement.get("quota_exhausted"):
            vision.update(quota_exhausted=True, status=refinement["status"])
        elif refinement["changes"]:
            context["parts"] = contextual_parts(context["parts"], refinement)
    gemini_seconds = perf_counter() - gemini_started
    if vision.get("quota_exhausted"):
        progress(0.56, GEMINI_QUOTA_NOTICE)
    observations = [] if vision.get("quota_exhausted") else vision["verified"]
    sources, count = contextual_sources(segments, context, observations)
    if not sources and vision.get("context_changes") and not vision.get("quota_exhausted"):
        sources = draft_sources(segments, speech_drafts(context["parts"]))
    note = "" if count else "Frame sudah dibaca. Belum ada kode yang bisa dikonfirmasi dan dipakai untuk menyusun langkah."
    visual = VisualProcessing(
        context["frame_count"], count, context["ocr_seconds"], note,
        gemini_model=vision.get("model", "") if vision.get("requests") else "",
        gemini_evidence=len(observations),
        gemini_seconds=gemini_seconds if vision.get("requests") else 0.0,
        gemini_context_changes=vision.get("context_changes", 0) if not vision.get("quota_exhausted") else 0,
        gemini_code_examples=tuple((item["code"], item["quote"]) for item in observations),
    )
    return sources, visual, vision


def finish_guide(
    video: bytes | str, segments: list[Segment], started: float, transcribed_in: float, progress: Progress,
    *, overview: bool, transcript_source: str | None = None, transcript_note: str = "",
    transcription_beam_size: int | None = None,
    visual_path: Path | None = None, youtube_scratch: Path | None = None,
) -> Guide:
    from src.summarization import model_reference

    checkpoint = model_reference()
    summary_started = perf_counter()
    overview_result = None
    preprocessing = "Lokal"
    preprocessing_note = ""
    visual = None
    quota = False
    status = coding_transcript_status(segments)
    recovered = []
    screen_sources = []
    if status in {"unclear", "out_of_scope"} and not overview:
        if is_string_lesson(segments):
            recovered = order_string_points(string_speech_points(segments))
        else:
            recovered = recover_tutorial_sources(segments)
        if recovered or is_string_lesson(segments):
            status = "ready"
    if status != "ready" and not overview and (visual_path is not None or youtube_scratch is not None):
        screen_sources, visual, _ = prepare_visual_sources(
            segments, video, visual_path, youtube_scratch, None, progress, screen_only=True,
        )
        if screen_sources:
            status = "ready"
    if status != "ready":
        preprocessing = "Dilewati"
        steps, reference = [], checkpoint
    elif overview:
        progress(0.58, "Menyiapkan teks dari seluruh video…")
        evidence = prepare_evidence(segments)
        if not evidence:
            raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")
        reference = checkpoint
        progress(0.66, "Membuat ringkasan keseluruhan…")
        tokenizer, model, reference = load_guide_model(reference)
        overview_result = summarize_overview(evidence, tokenizer, model)
        steps = []
    else:
        config = None
        if screen_sources:
            sources, vision = screen_sources, {}
        else:
            try:
                config = gemini_config()
            except GeminiError as error:
                preprocessing_note = f"{error} Aplikasi memakai preprocessing lokal."
            sources, visual, vision = prepare_visual_sources(segments, video, visual_path, youtube_scratch, config, progress)
        quota = bool(vision.get("quota_exhausted"))
        if quota:
            preprocessing_note = vision["status"]
        elif vision.get("failed"):
            preprocessing_note = vision["status"]
        if sources:
            if config and any(source.visual_quote for source in sources):
                preprocessing = f"Gemini multimodal · {config.model} + OCR"
            elif config and vision.get("context_changes"):
                preprocessing = f"Praproses kontekstual Gemini · {config.model}"
            else:
                preprocessing = "OCR lokal"
        elif config and not vision.get("failed") and not vision.get("requests"):
            try:
                progress(0.55, "Menyeleksi inti transkrip dengan Gemini…")
                sources = prepare_gemini_sources(segments, config)
                if sources:
                    preprocessing = f"Gemini · {config.model}"
                else:
                    preprocessing_note = "Belum ada cuplikan Gemini yang lolos pemeriksaan. Aplikasi memakai preprocessing lokal."
            except GeminiError as error:
                quota = isinstance(error, GeminiQuotaError)
                if quota:
                    progress(0.56, GEMINI_QUOTA_NOTICE)
                preprocessing_note = f"{error} Aplikasi memakai preprocessing lokal."
        if not sources and recovered:
            sources = recovered
            preprocessing = "Seleksi teknis lokal"
        if sources:
            steps, reference, overview_result = summarize_segments(segments, progress, sources=sources)
        else:
            steps, reference, overview_result = summarize_segments(segments, progress)
        if not any(step.summary for step in steps):
            candidates = recovered if is_string_lesson(segments) else (
                recovered or limit_tutorial_sources(recover_tutorial_sources(segments))
            )
            if candidates:
                steps = [Step(source.segment.start, source.segment.end, source.segment.transcript,
                              source.text, True, source.text) for source in candidates]
                preprocessing = "Seleksi teknis lokal"
            else:
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
        visual=visual, gemini_quota_exhausted=quota,
        model_setting=checkpoint,
    )
    progress(1.0, "Panduan siap." if status == "ready" else "Media selesai diperiksa.")
    return guide


def compare_transcript(guide: Guide, progress: Progress) -> tuple[str, ...]:
    from src.video_context import transcript_only_sources

    segments = list(guide.transcript)
    if not segments:
        return ()
    sources = transcript_only_sources(segments)
    steps, _, _ = summarize_segments(segments, progress, sources=sources, checkpoint=guide.model_setting)
    return tuple(dict.fromkeys(step.summary for step in steps if step.summary))


def retry_sparse_transcript(media: Path, segments: list[Segment], progress: Progress) -> list[Segment]:
    if (len(segments) < 6 or coding_transcript_status(segments) != "ready"
            or recover_tutorial_sources(segments) or any(source.text for source in prepare_step_sources(segments))):
        return segments
    progress(0.42, "Memeriksa ulang audio untuk menemukan instruksi teknis…")
    try:
        revised = transcribe(media, retry_noisy=True)
    except (MediaError, OSError, RuntimeError, ValueError):
        log.info("Pemeriksaan ulang transkrip gagal.")
        return segments
    return revised if recover_tutorial_sources(revised) else segments


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
        revised = segments if overview else retry_sparse_transcript(playable, segments, progress)
        transcribed_in = perf_counter() - transcription_started
        playback = video if playable == upload else playable.read_bytes()
        return finish_guide(
            playback, revised, started, transcribed_in, progress, overview=overview, visual_path=playable,
            transcription_beam_size=max(3, whisper_beam_size()) if revised is not segments else None,
            transcript_note="Bagian teknis ditemukan setelah audio diperiksa ulang." if revised is not segments else "",
        )


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
            if captions_need_review(segments):
                caption_coverage = readable_coverage(segments)
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
            revised = segments if overview else retry_sparse_transcript(audio, segments, progress)
            if revised is not segments:
                segments = revised
                corrected_beam = max(3, whisper_beam_size())
                transcript_note = "Bagian teknis ditemukan setelah audio diperiksa ulang."
        transcribed_in = perf_counter() - transcription_started + inspection_time
        return finish_guide(
            video, segments, started, transcribed_in, progress, overview=overview, transcript_source=transcript_source,
            transcript_note=transcript_note, transcription_beam_size=corrected_beam, youtube_scratch=Path(scratch),
        )
