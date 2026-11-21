import logging

import streamlit as st

from src.pipeline import Guide, build_guide


log = logging.getLogger(__name__)


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    return f"{hours:d}:{clock}" if hours else clock


def reset_guide() -> None:
    for key in ("guide", "video_name", "selected_step", "seek_revision"):
        st.session_state.pop(key, None)


def select_step(index: int) -> None:
    st.session_state.selected_step = index
    st.session_state.seek_revision += 1


def upload_view() -> None:
    st.title("Ubah tutorial Python jadi panduan praktik")
    st.write("Unggah tutorial coding berbahasa Indonesia untuk menemukan langkah yang ingin kamu praktikkan.")

    with st.container(border=True):
        video = st.file_uploader("Video tutorial", type=["mp4"], help="Pilih MP4 dengan video dan narasi.")
        make_guide = st.button("Buat Panduan", type="primary", disabled=video is None)
    if not make_guide or video is None:
        return

    progress = st.progress(0.0, text="Menyiapkan video…")
    try:
        guide = build_guide(video.getvalue(), lambda value, text: progress.progress(value, text=text))
    except ValueError as error:
        progress.empty()
        log.info("Panduan gagal dibuat", exc_info=True)
        st.error(str(error))
        return
    except (OSError, RuntimeError):
        progress.empty()
        log.exception("Pemrosesan video gagal")
        st.error("Panduan gagal diproses. Periksa koneksi saat model diunduh pertama kali dan lihat detail di PowerShell.")
        return

    st.session_state.guide = guide
    st.session_state.video_name = video.name
    st.session_state.selected_step = None
    st.session_state.seek_revision = 0
    st.rerun()


def guide_view(guide: Guide) -> None:
    title, controls = st.columns([3, 1], vertical_alignment="center")
    with title:
        st.title("Panduan praktik")
        st.text(st.session_state.video_name)
    with controls:
        st.button("Ganti video", on_click=reset_guide)

    player, steps = st.columns([3, 2], gap="large")
    selected = st.session_state.selected_step
    start = guide.steps[selected].start if selected is not None else 0
    with player:
        # Remount the video on every seek, including a second click at the same time.
        with st.container(key=f"player-{st.session_state.seek_revision % 2}"):
            st.video(guide.video, start_time=int(start), autoplay=selected is not None)
        st.caption(f"Whisper base · {guide.model}")

    with steps:
        st.subheader("Langkah praktik")
        for index, step in enumerate(guide.steps):
            with st.container(border=True):
                st.button(
                    timestamp(step.start),
                    key=f"seek-{index}",
                    help=f"Putar dari awal langkah {index + 1}",
                    type="primary" if selected == index else "secondary",
                    on_click=select_step,
                    args=(index,),
                )
                if step.is_excerpt:
                    st.caption("Cuplikan sumber")
                st.text(step.summary)
                with st.expander("Lihat transkrip"):
                    st.caption(f"{timestamp(step.start)}–{timestamp(step.end)}")
                    st.text(step.transcript)


st.set_page_config(page_title="Panduan Praktik Python", layout="wide")
if "guide" in st.session_state:
    guide_view(st.session_state.guide)
else:
    upload_view()
