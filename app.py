from collections.abc import Callable
from html import escape
import logging
from pathlib import Path

import streamlit as st

from src.media import MediaError
from src.pipeline import Guide, Progress, Step, build_guide, build_youtube_guide


log = logging.getLogger(__name__)
STYLES = Path(__file__).with_name("assets") / "app.css"


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    return f"{hours:d}:{clock}" if hours else clock


def reset_guide() -> None:
    for key in ("guide", "video_name", "selected_step", "seek_revision", "step-search"):
        st.session_state.pop(key, None)


def select_step(index: int) -> None:
    st.session_state.selected_step = index
    st.session_state.seek_revision += 1


def process_guide(name: str, build: Callable[[Progress], Guide]) -> None:
    progress = st.progress(0.0, text="Menyiapkan panduan…")
    try:
        guide = build(lambda value, text: progress.progress(value, text=text))
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
    st.session_state.selected_step = None
    st.session_state.seek_revision = 0
    st.rerun()


def upload_view() -> None:
    with st.container(key="page-heading"):
        st.title("Buat panduan praktik")
        st.write("Ubah video tutorial menjadi langkah yang bisa diputar dari penanda waktu.")

    with st.container(key="upload-layout"):
        sources, notes = st.columns([5, 3], gap="large")
        with sources:
            upload_tab, youtube_tab = st.tabs(["Unggah MP4", "Link YouTube"])
            with upload_tab, st.container(key="upload-panel", border=True):
                video = st.file_uploader("Video tutorial", type=["mp4"], key="video-upload")
                make_guide = st.button(
                    "Buat Panduan", type="primary", key="make-mp4-guide", disabled=video is None,
                )
                if make_guide and video is not None:
                    process_guide(video.name, lambda progress: build_guide(video.getvalue(), progress))

            with youtube_tab, st.container(key="youtube-panel", border=True):
                link = st.text_input(
                    "Link video YouTube", key="youtube-link", placeholder="https://www.youtube.com/watch?v=…",
                )
                st.caption("Whisper akan membuat transkrip dari audio video YouTube.")
                make_guide = st.button(
                    "Buat Panduan", type="primary", key="make-youtube-guide",
                    disabled=not link.strip(),
                )
                if make_guide and link.strip():
                    process_guide("YouTube", lambda progress: build_youtube_guide(link, progress))
        with notes, st.container(key="upload-notes"):
            st.subheader("Sebelum mulai")
            st.markdown(
                "- Pilih tutorial Python dengan narasi bahasa Indonesia yang jelas.\n"
                "- Untuk uji pertama, gunakan klip pendek. Pemuatan model pertama bisa lebih lama."
            )
            st.caption("Video dan panduan tersedia selama sesi ini.")


def show_step(index: int, step: Step, *, active: bool) -> None:
    row_state = "active" if active else "idle"
    with st.container(key=f"step-row-{row_state}-{index}"):
        time_column, content = st.columns([1, 5], gap="small")
        with time_column:
            st.button(
                timestamp(step.start),
                key=f"seek-{index}",
                help=f"Putar dari awal langkah {index + 1}",
                type="primary" if active else "secondary",
                on_click=select_step,
                args=(index,),
            )
        with content:
            st.html(f'<p class="step-summary">{escape(step.summary)}</p>')
            if step.is_excerpt:
                st.caption("Cuplikan sumber")
            with st.expander("Lihat transkrip"):
                st.caption(f"{timestamp(step.start)}–{timestamp(step.end)}")
                st.html(f'<p class="source-text">{escape(step.transcript.strip())}</p>')


def guide_view(guide: Guide) -> None:
    with st.container(key="guide-heading"):
        title, controls = st.columns([4, 1], vertical_alignment="center")
        with title:
            st.title("Panduan praktik")
            st.html(f'<p class="video-name">{escape(st.session_state.video_name)}</p>')
        with controls:
            st.button("Ganti video", on_click=reset_guide)

    with st.container(key="guide-layout"):
        player, steps = st.columns([3, 2], gap="large")
        selected = st.session_state.selected_step
        start = guide.steps[selected].start if selected is not None else 0
        with player, st.container(key="video-pane"):
            st.subheader("Video")
            # Remount the video on every seek, including a second click at the same time.
            with st.container(key=f"player-{st.session_state.seek_revision % 2}"):
                st.video(guide.video, start_time=int(start), autoplay=selected is not None)
            if isinstance(guide.video, str):
                st.link_button("Buka di YouTube", f"{guide.video}&t={int(start)}s")
            with st.expander("Detail pemrosesan"):
                st.text(f"Whisper {guide.transcription_model}\n{guide.model}")
                st.caption(f"Waktu proses: {guide.processing_seconds:.1f} detik")

        with steps, st.container(key="steps-pane"):
            st.subheader("Langkah praktik")
            query = st.text_input(
                "Cari langkah", key="step-search", placeholder="Contoh: dropna atau harga kosong",
                help="Ketik kata dari ringkasan atau transkrip, lalu tekan Enter.",
            )
            matches = guide.find_steps(query)
            count = f"{len(matches)} dari {len(guide.steps)}" if query.strip() else str(len(guide.steps))
            st.caption(f"{count} langkah")
            if not matches:
                st.info("Langkah tidak ditemukan. Coba kata lain atau kosongkan pencarian.")
            for index, step in matches:
                show_step(index, step, active=selected == index)


st.set_page_config(page_title="Panduan Praktik Python", layout="wide")
st.html(STYLES)
st.html(
    '<div class="app-masthead"><span class="app-glyph" aria-hidden="true">&gt;_</span>'
    '<span>Panduan Python</span></div>'
)
if "guide" in st.session_state:
    guide_view(st.session_state.guide)
else:
    upload_view()
