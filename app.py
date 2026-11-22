from collections.abc import Callable
from html import escape
import logging
from pathlib import Path

import streamlit as st

from src.evaluation import normalize, speech_metrics, summary_metrics
from src.media import MediaError
from src.pipeline import Guide, Progress, build_guide, build_youtube_guide


log = logging.getLogger(__name__)
STYLES = Path(__file__).with_name("assets") / "app.css"


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    return f"{hours:d}:{clock}" if hours else clock


def reset_guide() -> None:
    for key in ("guide", "video_name", "seek_start", "seek_revision", "guide-evaluation"):
        st.session_state.pop(key, None)


def seek(seconds: float) -> None:
    st.session_state.seek_start = seconds
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
    st.session_state.seek_start = None
    st.session_state.seek_revision = 0
    st.rerun()


def upload_view() -> None:
    with st.container(key="page-heading"):
        st.title("Buat panduan praktik")
        st.write("Buat panduan langkah demi langkah dari video tutorial berbahasa Indonesia.")

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
                st.caption("Aplikasi mengambil subtitle Indonesia jika tersedia atau mentranskripsi audio dengan Whisper.")
                make_guide = st.button(
                    "Buat Panduan", type="primary", key="make-youtube-guide",
                    disabled=not link.strip(),
                )
                if make_guide and link.strip():
                    process_guide("YouTube", lambda progress: build_youtube_guide(link, progress, prefer_captions=True))
        with notes, st.container(key="upload-notes"):
            st.subheader("Sebelum mulai")
            st.markdown(
                "- Pilih tutorial Python dengan narasi bahasa Indonesia yang jelas.\n"
                "- Untuk uji pertama, gunakan klip pendek. Pemuatan model pertama bisa lebih lama."
            )
            st.caption("Video dan panduan tersedia selama sesi ini.")


def show_evaluation(transcript: str, summary: str) -> None:
    with st.expander("Evaluasi dengan teks acuan"):
        st.caption("WER dan CER memakai transkrip acuan. ROUGE-1, ROUGE-2, dan ROUGE-L memakai ringkasan acuan.")
        with st.form("reference-evaluation", border=False):
            transcript_reference = st.text_area("Transkrip acuan", placeholder="Tempel transkrip yang sudah diperiksa.")
            summary_reference = st.text_area("Ringkasan acuan", placeholder="Tempel ringkasan referensi.")
            calculate = st.form_submit_button("Hitung skor")
        if calculate:
            scores = {}
            if normalize(transcript_reference):
                scores.update(speech_metrics([transcript_reference], [transcript]))
            if normalize(summary_reference):
                scores.update(summary_metrics(summary_reference, summary))
            st.session_state["guide-evaluation"] = scores
        scores = st.session_state.get("guide-evaluation") or {}
        if scores:
            labels = {
                "wer": "WER", "cer": "CER", "rouge1": "ROUGE-1",
                "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
            }
            st.markdown(
                "| Metrik | Skor |\n| :-- | --: |\n"
                + "\n".join(f"| {labels[name]} | {score:.2%} |" for name, score in scores.items())
            )
        else:
            st.caption("Skor tersedia setelah teks acuan diisi.")


def show_analysis(guide: Guide) -> None:
    import altair as alt

    transcript = " ".join(segment.text for segment in guide.transcript)
    summary = " ".join(step.summary for step in guide.steps)
    engine = "faster-whisper · CPU INT8" if guide.transcription_backend == "faster" else "OpenAI Whisper · CPU"
    source = guide.transcript_source or f"Whisper {guide.transcription_model} · {engine}"
    media = "YouTube" if isinstance(guide.video, str) else "MP4 tervalidasi"
    outputs = (
        ("Sumber video", media),
        ("Transkrip", f"{len(guide.transcript)} segmen · {len(transcript.split())} kata"),
        ("Teks terpilih", f"{len(guide.steps)} bagian untuk mBART"),
        ("Panduan", f"{len(guide.steps)} langkah · {len(summary.split())} kata"),
    )
    with st.container(key="analysis-panel"):
        st.subheader("Analisis pemrosesan")
        st.html(
            '<ol class="pipeline-flow" aria-label="Alur pemrosesan">'
            + "".join(
                f'<li><strong>{escape(label)}</strong><span>{escape(value)}</span></li>'
                for label, value in outputs
            )
            + "</ol>"
        )
        if guide.timings is not None:
            st.markdown("**Waktu per tahap**")
            stages = ["Media", "Transkrip", "Ringkasan"]
            values = [guide.timings.media, guide.timings.transcription, guide.timings.summarization]
            base = alt.Chart(alt.Data(values=[
                {"Tahap": stage, "Detik": round(duration, 2)} for stage, duration in zip(stages, values, strict=True)
            ])).encode(
                y=alt.Y("Tahap:N", sort=stages, title=None, axis=alt.Axis(labelOverlap=False, labelPadding=8)),
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
            st.markdown("**Media**")
            st.text(guide.video if isinstance(guide.video, str) else f"MP4 siap diputar · {len(guide.video) / 1_000_000:.2f} MB")
            st.markdown("**Transkrip**")
            st.caption(source)
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
                f"<li>{timestamp(step.start)} · {escape(step.model_input)}</li>" for step in guide.steps
            ) + "</ol>")
            st.caption(f"Model ringkasan {guide.model}")
            excerpts = sum(step.is_excerpt for step in guide.steps)
            if excerpts:
                st.caption(f"{excerpts} dari {len(guide.steps)} langkah memakai teks sumber setelah pemeriksaan keluaran mBART.")
        show_evaluation(transcript, summary)


def guide_view(guide: Guide) -> None:
    if not guide.steps:
        reset_guide()
        st.rerun()
        return
    with st.container(key="guide-heading"):
        title, controls = st.columns([4, 1], vertical_alignment="center")
        with title:
            st.title("Panduan video")
            st.html(f'<p class="video-name">{escape(st.session_state.video_name)}</p>')
        with controls:
            st.button("Ganti video", on_click=reset_guide)

    with st.container(key="guide-layout"):
        player, summary = st.columns([3, 2], gap="large")
        selected = st.session_state.seek_start
        start = selected if selected is not None else 0
        with player, st.container(key="video-pane"):
            st.subheader("Video")
            with st.container(key=f"player-{st.session_state.seek_revision % 2}"):
                st.video(guide.video, start_time=int(start), autoplay=selected is not None)
            if isinstance(guide.video, str):
                st.link_button("Buka di YouTube", f"{guide.video}&t={int(start)}s")
        with summary, st.container(key="summary-pane"):
            st.subheader("Langkah dalam video")
            with st.container(key="timeline"):
                for index, step in enumerate(guide.steps):
                    clock, action = st.columns([1, 5], gap="small")
                    with clock:
                        st.button(
                            timestamp(step.start), key=f"seek-step-{index}", type="tertiary",
                            help=f"Putar bagian {timestamp(step.start)}–{timestamp(step.end)}",
                            on_click=seek, args=(step.start,),
                        )
                    with action:
                        st.html(f'<p class="step-copy">{escape(step.summary)}</p>')
    show_analysis(guide)


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
