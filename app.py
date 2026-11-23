from collections.abc import Callable
from html import escape
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.code_text import highlight_code
from src.gemini_preprocessing import GEMINI_QUOTA_NOTICE
from src.media import MediaError
from src.pipeline import Guide, Progress, YouTubeInspection, build_youtube_guide, inspect_youtube


log = logging.getLogger(__name__)
STYLES = Path(__file__).with_name("assets") / "app.css"
UNREADABLE_STEP = "Ucapan di bagian ini belum cukup jelas untuk diringkas."
UNAVAILABLE_SUMMARY = {
    "no_speech": "Narasi tidak terdeteksi. Video tetap bisa diputar untuk memeriksa audionya.",
    "out_of_scope": "Transkripnya tidak menunjukkan materi pemrograman yang cukup untuk dirangkum.",
    "unclear": "Belum ada bagian teknis yang bisa dijadikan poin. Periksa transkrip dan pembacaan layar di Hasil tiap tahap.",
}


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    return f"{hours:d}:{clock}" if hours else clock


def reset_guide() -> None:
    for key in ("guide", "video_name", "seek_start", "seek_revision", "pending-youtube", "youtube-link", "youtube-cancelled"):
        st.session_state.pop(key, None)


def seek(seconds: float) -> None:
    st.session_state.seek_start = seconds
    st.session_state.seek_revision += 1


def process_guide(name: str, build: Callable[[Progress], Guide]) -> None:
    progress = st.progress(0.0, text="Menyiapkan panduan…")
    notice = st.empty()

    def report_progress(value: float, text: str) -> None:
        if text == GEMINI_QUOTA_NOTICE:
            notice.warning(text)
        else:
            progress.progress(value, text=text)

    try:
        guide = build(report_progress)
    except MediaError as error:
        progress.empty()
        log.info("Panduan tidak dapat diproses: %s", error)
        st.error(str(error))
        return
    except (OSError, RuntimeError, ValueError, ImportError):
        progress.empty()
        log.exception("Pemrosesan panduan gagal")
        st.error("Panduan gagal dibuat. Coba ulangi dan periksa detailnya di terminal Streamlit.")
        return

    st.session_state.guide = guide
    st.session_state.video_name = name
    st.session_state.seek_start = None
    st.session_state.seek_revision = 0
    st.rerun()


def youtube_view() -> None:
    with st.container(key="page-heading"):
        st.title("Ringkas video tutorial")
        st.write("Ringkas tutorial pemrograman berbahasa Indonesia dari YouTube.")

    with st.container(key="youtube-panel", border=True):
        show_youtube_input()


def cancel_youtube_check() -> None:
    st.session_state.pop("pending-youtube", None)
    st.session_state["youtube-cancelled"] = True


@st.dialog("Periksa sumber transkrip")
def confirm_youtube_processing(inspection: YouTubeInspection) -> None:
    if inspection.captions is None:
        st.warning("Subtitle bahasa Indonesia belum ditemukan. Whisper bisa mentranskripsi audio, tetapi nama fungsi atau angka mungkin salah terbaca.")
    else:
        st.warning("Subtitle bahasa Indonesia terdeteksi, tetapi teksnya sulit dibaca. Jika dilanjutkan, aplikasi akan membandingkannya dengan transkrip Whisper sebelum membuat ringkasan.")
    actions = st.empty()
    with actions.container():
        with st.container(key="youtube-dialog-actions"):
            cancel, proceed = st.columns(2, gap="small")
            cancel_clicked = cancel.button("Batal")
            proceed_clicked = proceed.button("Lanjut dengan Whisper", type="primary")
    if cancel_clicked:
        cancel_youtube_check()
        st.rerun()
    if proceed_clicked:
        st.session_state.pop("pending-youtube", None)
        actions.empty()
        with actions.container():
            process_guide("YouTube", lambda progress: build_youtube_guide(inspection, progress))


def show_youtube_input() -> None:
    link = st.text_input("Link video YouTube", key="youtube-link", placeholder="https://www.youtube.com/watch?v=…").strip()
    st.caption("Subtitle bahasa Indonesia akan diperiksa sebelum pemrosesan.")
    if st.session_state.pop("youtube-cancelled", False):
        st.info("Pemrosesan dibatalkan.")
    pending = st.session_state.get("pending-youtube")
    if pending is not None and pending[0] != link:
        st.session_state.pop("pending-youtube")
        pending = None
    if st.button("Buat Panduan", type="primary", key="make-youtube-guide", disabled=not link) and pending is None:
        try:
            with st.spinner("Memeriksa subtitle bahasa Indonesia…"):
                inspection = inspect_youtube(link)
        except MediaError as error:
            st.error(str(error))
        else:
            if inspection.captions is not None and not inspection.captions_need_review:
                _, subtitle_source = inspection.captions
                st.toast(f"{subtitle_source} terdeteksi. Aplikasi akan memeriksa kualitasnya sebelum merangkum.")
                process_guide("YouTube", lambda progress: build_youtube_guide(inspection, progress))
            else:
                st.session_state["pending-youtube"] = (link, inspection)
                st.rerun()
    if pending is not None:
        confirm_youtube_processing(pending[1])


def show_analysis(guide: Guide) -> None:
    transcript = " ".join(segment.text for segment in guide.transcript)
    engine = "faster-whisper · CPU INT8" if guide.transcription_backend == "faster" else "OpenAI Whisper · CPU"
    source = guide.transcript_source or f"Whisper {Path(guide.transcription_model).name} · {engine} · {guide.transcription_beam_size} beam"
    preprocessing = getattr(guide, "preprocessing", "Tidak tercatat")
    visual = getattr(guide, "visual", None)
    gemini = getattr(visual, "gemini_model", "")
    outputs = (
        ("Transkrip", f"{len(guide.transcript)} segmen · {len(transcript.split())} kata"),
        ("Praproses", f"{len(guide.steps)} bagian dipilih"),
        ("OCR lokal", f"{visual.frames} frame diperiksa") if visual and visual.frames else None,
        ("Gemini", f"{gemini} · {visual.gemini_evidence} bukti terverifikasi") if gemini else None,
        ("Peringkasan", f"{len(guide.summary_points)} poin · {preprocessing}"),
    )
    with st.container(key="analysis-panel"):
        st.subheader("Analisis pemrosesan")
        st.html(
            '<ol class="pipeline-flow" aria-label="Alur pemrosesan">'
            + "".join(
                f'<li><strong>{escape(label)}</strong><span>{escape(value)}</span></li>'
                for label, value in (item for item in outputs if item)
            )
            + "</ol>"
        )
        show_pipeline_details(guide, source)


def show_pipeline_details(guide: Guide, source: str) -> None:
    import altair as alt

    preprocessing_note = getattr(guide, "preprocessing_note", "")
    if preprocessing_note:
        st.caption(preprocessing_note)
    visual = getattr(guide, "visual", None)
    if visual and visual.note:
        st.caption(visual.note)
    if guide.timings is not None:
        st.markdown("**Waktu per tahap**")
        stages = ["Media", "Transkrip", "Preprocessing dan ringkasan"]
        values = [guide.timings.media, guide.timings.transcription, guide.timings.summarization]
        if visual and visual.frames:
            gemini_seconds = getattr(visual, "gemini_seconds", 0)
            stages = [*stages[:2], "Ekstraksi frame dan OCR"]
            values = [*values[:2], visual.seconds]
            if gemini_seconds:
                stages.append("Gemini")
                values.append(gemini_seconds)
            stages.append("Penyusunan dan ringkasan")
            values.append(max(0, guide.timings.summarization - visual.seconds - gemini_seconds))
        base = alt.Chart(pd.DataFrame([
            {"Tahap": stage, "Detik": round(duration, 2)} for stage, duration in zip(stages, values, strict=True)
        ])).encode(
            y=alt.Y("Tahap:N", sort=stages, title=None, axis=alt.Axis(labelOverlap=False, labelPadding=8, labelLimit=160)),
            x=alt.X(
                "Detik:Q", title="Detik", axis=alt.Axis(tickCount=5),
                scale=alt.Scale(domain=[0, max(1.0, max(values) * 1.22)]),
            ),
        )
        bars = base.mark_bar(color="#2858B8", size=14).encode(
            tooltip=["Tahap:N", alt.Tooltip("Detik:Q", format=".2f")],
        )
        labels = base.mark_text(align="left", dx=5, color="#202734").encode(text=alt.Text("Detik:Q", format=".1f"))
        chart = (bars + labels).properties(height=175)
        st.altair_chart(chart)
        st.caption(f"Total {guide.processing_seconds:.1f} detik · {source}")

    with st.expander("Hasil tiap tahap"):
        st.markdown("**Transkrip**")
        st.caption(source)
        if guide.transcript_note:
            st.caption(guide.transcript_note)
        segment_index = st.selectbox(
            "Segmen untuk diputar", range(len(guide.transcript)),
            format_func=lambda index: (
                f"{timestamp(guide.transcript[index].start)} · {guide.transcript[index].text[:64]}"
            ),
        )
        if segment_index is not None:
            st.button("Putar segmen", on_click=seek, args=(guide.transcript[segment_index].start,))
        st.html(
            '<div class="transcript-lines" tabindex="0" aria-label="Transkrip bertanda waktu">'
            + "".join(
                f'<p><span>{timestamp(segment.start)}</span> {escape(segment.text)}</p>'
                for segment in guide.transcript
            )
            + "</div>"
        )
        st.markdown("**Teks yang dipilih**")
        st.html("<ol>" + "".join(
            f"<li>{timestamp(step.start)} · {escape(step.model_input or UNREADABLE_STEP)}</li>" for step in guide.steps
        ) + "</ol>")
        if any(step.model_input for step in guide.steps):
            st.caption(f"Model ringkasan {guide.model}")
        excerpts = sum(step.is_excerpt for step in guide.steps)
        if excerpts:
            st.caption(f"{excerpts} dari {len(guide.steps)} langkah memakai teks sumber setelah pemeriksaan keluaran model.")


def guide_view(guide: Guide) -> None:
    with st.container(key="guide-heading"):
        title, controls = st.columns([4, 1], vertical_alignment="center")
        with title:
            st.title("Ringkasan video")
            st.html(f'<p class="video-name">{escape(st.session_state.video_name)}</p>')
        with controls:
            st.button("Ganti video", on_click=reset_guide)

    if getattr(guide, "gemini_quota_exhausted", False):
        st.warning(GEMINI_QUOTA_NOTICE)

    with st.container(key="guide-layout"):
        player, summary = st.columns([3, 2], gap="large")
        selected = st.session_state.seek_start
        start = selected if selected is not None else 0
        with player, st.container(key="video-pane"):
            st.subheader("Video")
            with st.container(key=f"player-{st.session_state.seek_revision % 2}"):
                st.video(guide.video, start_time=int(start), autoplay=selected is not None)
            st.link_button("Buka di YouTube", f"{guide.video}&t={int(start)}s")
        with summary, st.container(key="summary-pane"):
            if guide.overview and guide.overview.text:
                st.subheader("Tentang video")
                st.html(f'<p class="video-intro">{highlight_code(guide.overview.text)}</p>')
            st.subheader("Poin utama")
            if guide.summary_points:
                st.html('<ul class="summary-points">' + "".join(
                    f"<li>{highlight_code(point)}</li>" for point in guide.summary_points
                ) + "</ul>")
            else:
                st.info(UNAVAILABLE_SUMMARY.get(guide.status, UNAVAILABLE_SUMMARY["unclear"]))
    show_analysis(guide)


st.set_page_config(page_title="Panduan Pemrograman", layout="wide")
st.html(STYLES)
st.html(
    '<div class="app-masthead"><span class="app-glyph" aria-hidden="true">&gt;_</span>'
    '<span>Panduan Pemrograman</span></div>'
)
active_guide = st.session_state.get("guide")
if active_guide is not None and isinstance(active_guide.video, str):
    guide_view(active_guide)
else:
    if active_guide is not None:
        reset_guide()
    youtube_view()
