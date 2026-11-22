from collections.abc import Callable
from html import escape
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.code_text import highlight_code, technical_terms
from src.evaluation import summary_metrics
from src.evaluation_view import show_evaluation
from src.media import MediaError
from src.pipeline import Guide, Progress, build_guide, build_youtube_guide, inspect_youtube


log = logging.getLogger(__name__)
STYLES = Path(__file__).with_name("assets") / "app.css"
UNREADABLE_STEP = "Ucapan di bagian ini belum cukup jelas untuk diringkas."
UNAVAILABLE_SUMMARY = {
    "no_speech": "Narasi tidak terdeteksi. Video tetap bisa diputar untuk memeriksa audionya.",
    "out_of_scope": "Transkripnya tidak menunjukkan materi pemrograman yang cukup untuk dirangkum.",
    "unclear": "Topik atau penjelasannya belum cukup jelas untuk dibuat ringkasan yang akurat.",
}


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes:02d}:{seconds:02d}"
    return f"{hours:d}:{clock}" if hours else clock


def reset_guide() -> None:
    for key in ("guide", "video_name", "seek_start", "seek_revision", "guide-evaluation", "step-reviews", "audit-step", "pending-youtube"):
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
        st.title("Ringkas video tutorial")
        st.write("Dapatkan poin utama dan hasil percobaan dari tutorial pemrograman berbahasa Indonesia.")

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
                show_youtube_input()
        with notes, st.container(key="upload-notes"):
            st.subheader("Sebelum mulai")
            st.markdown(
                "- Pilih tutorial pemrograman dengan narasi bahasa Indonesia yang jelas.\n"
                "- Untuk uji pertama, gunakan klip pendek. Pemuatan model pertama bisa lebih lama."
            )
            st.caption("Video dan panduan tersedia selama sesi ini.")


def cancel_youtube_check() -> None:
    st.session_state.pop("pending-youtube", None)
    st.session_state["youtube-cancelled"] = True


def show_youtube_input() -> None:
    link = st.text_input("Link video YouTube", key="youtube-link", placeholder="https://www.youtube.com/watch?v=…").strip()
    st.caption("Subtitle bahasa Indonesia akan diperiksa sebelum pemrosesan.")
    if st.session_state.pop("youtube-cancelled", False):
        st.info("Pemrosesan dibatalkan.")
    pending = st.session_state.get("pending-youtube")
    if pending is not None and pending[0] != link:
        st.session_state.pop("pending-youtube")
        pending = None
    if st.button("Buat Panduan", type="primary", key="make-youtube-guide", disabled=not link or pending is not None):
        try:
            with st.spinner("Memeriksa subtitle bahasa Indonesia…"):
                inspection = inspect_youtube(link)
        except MediaError as error:
            st.error(str(error))
        else:
            if inspection.captions is not None:
                process_guide("YouTube", lambda progress: build_youtube_guide(inspection, progress))
            else:
                st.session_state["pending-youtube"] = (link, inspection)
                st.rerun()
    if pending is not None:
        st.warning("Subtitle bahasa Indonesia belum ditemukan. Whisper bisa mentranskripsi audio, tetapi nama fungsi atau angka mungkin salah terbaca.")
        proceed, cancel = st.columns([2, 1], gap="small")
        if proceed.button("Lanjut dengan Whisper", type="primary"):
            st.session_state.pop("pending-youtube")
            process_guide("YouTube", lambda progress: build_youtube_guide(pending[1], progress))
        cancel.button("Batal", on_click=cancel_youtube_check)


def show_step_audit(guide: Guide) -> None:
    if not guide.steps:
        st.caption("Audit tersedia setelah aplikasi menemukan bagian tutorial yang bisa diperiksa.")
        return
    st.caption("Bandingkan ringkasan dengan masukan mBART dan transkrip aslinya, lalu catat penilaianmu per langkah.")
    index = st.selectbox(
        "Langkah yang diperiksa", range(len(guide.steps)), key="audit-step",
        format_func=lambda index: f"{timestamp(guide.steps[index].start)} · {(guide.steps[index].summary or UNREADABLE_STEP)[:75]}",
    )
    step = guide.steps[index]
    st.button("Putar langkah ini", key="audit-seek", on_click=seek, args=(step.start,))
    original, rendered = st.columns(2)
    with original:
        st.markdown("**Teks masukan mBART**")
        st.html(f'<p class="step-copy">{highlight_code(step.model_input)}</p>')
    with rendered:
        st.markdown("**Ringkasan langkah**")
        st.html(f'<p class="step-copy">{highlight_code(step.summary or UNREADABLE_STEP)}</p>')
        if step.summary:
            st.caption("Disusun dari teks sumber" if step.is_excerpt else "Keluaran mBART")

    source_terms = technical_terms(step.model_input) if step.summary else set()
    missing = sorted(source_terms - technical_terms(step.summary))
    if source_terms:
        st.progress(
            (len(source_terms) - len(missing)) / len(source_terms),
            text=f"Istilah teknis yang muncul kembali · {len(source_terms) - len(missing)} dari {len(source_terms)}",
        )
        if missing:
            st.text("Tidak muncul dalam ringkasan · " + ", ".join(missing))
    if step.model_input and step.summary:
        overlap = summary_metrics(step.model_input, step.summary)["rougeL"]
        st.caption(f"Kemiripan urutan kata terhadap masukan mBART · {overlap:.1%} (ROUGE-L F1).")
    with st.expander("Lihat transkrip sumber"):
        st.text(step.transcript)

    aspects = ("Kejelasan ringkasan", "Informasi penting", "Ketepatan pemilihan segmen")
    ratings = ["Sesuai", "Perlu perbaikan"]
    reviews = st.session_state.setdefault("step-reviews", {})
    saved = reviews.get(index, {})
    with st.form(f"review-step-{index}", border=False):
        answers = {
            aspect: st.radio(
                aspect, ratings, index=ratings.index(saved[aspect]) if saved.get(aspect) in ratings else None,
                horizontal=True,
            ) for aspect in aspects
        }
        note = st.text_area("Catatan kesalahan", value=saved.get("note", ""), placeholder="Contoh: fungsi disebut salah atau konteksnya hilang.")
        if st.form_submit_button("Simpan penilaian") and (any(answers.values()) or note.strip()):
            reviews[index] = {**answers, "note": note}
            st.caption("Penilaian tersimpan selama sesi ini.")
    if reviews:
        st.caption(f"Langkah yang sudah ditinjau · {len(reviews)} dari {len(guide.steps)}")


def show_analysis(guide: Guide) -> None:
    transcript = " ".join(segment.text for segment in guide.transcript)
    summary = " ".join(guide.summary_points)
    model_input = " ".join(step.model_input for step in guide.steps)
    step_count = sum(bool(step.summary) for step in guide.steps)
    engine = "faster-whisper · CPU INT8" if guide.transcription_backend == "faster" else "OpenAI Whisper · CPU"
    source = guide.transcript_source or f"Whisper {Path(guide.transcription_model).name} · {engine} · {guide.transcription_beam_size} beam"
    media = "YouTube" if isinstance(guide.video, str) else "MP4 tervalidasi"
    outputs = (
        ("Sumber video", media),
        ("Transkrip", f"{len(guide.transcript)} segmen · {len(transcript.split())} kata"),
        ("Preprocessing", f"{guide.preprocessing} · {len(guide.steps)} bagian"),
        ("mBART-50", f"{len(model_input.split())} kata sebagai masukan" if model_input.strip() else "Tidak ada masukan yang layak"),
        ("Ringkasan", f"{step_count} segmen diringkas" if step_count else "Tidak tersedia"),
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
        flow, scores, audit = st.tabs(["Alur dan waktu", "Skor acuan", "Audit langkah"])
        with scores:
            show_evaluation(transcript, summary, model_input)
        with audit:
            show_step_audit(guide)
        with flow:
            show_pipeline_details(guide, source)


def show_pipeline_details(guide: Guide, source: str) -> None:
    import altair as alt

    if guide.preprocessing_note:
        st.caption(guide.preprocessing_note)
    if guide.timings is not None:
        st.markdown("**Waktu per tahap**")
        stages = ["Media", "Transkrip", "Preprocessing dan ringkasan"]
        values = [guide.timings.media, guide.timings.transcription, guide.timings.summarization]
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
        st.markdown("**Media**")
        st.text(guide.video if isinstance(guide.video, str) else f"MP4 siap diputar · {len(guide.video) / 1_000_000:.2f} MB")
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
            st.caption(f"{excerpts} dari {len(guide.steps)} langkah memakai teks sumber setelah pemeriksaan keluaran mBART.")


def guide_view(guide: Guide) -> None:
    with st.container(key="guide-heading"):
        title, controls = st.columns([4, 1], vertical_alignment="center")
        with title:
            st.title("Ringkasan video")
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
            if guide.overview and guide.overview.text:
                st.subheader("Tentang video")
                st.html(f'<p class="video-intro">{highlight_code(guide.overview.text)}</p>')
            st.subheader("Poin utama")
            if guide.summary_points:
                st.html('<ul class="summary-points">' + "".join(
                    f"<li>{highlight_code(point)}</li>" for point in guide.summary_points
                ) + "</ul>")
                if any(not step.summary for step in guide.steps):
                    st.caption("Segmen yang kurang jelas bisa diperiksa di Audit langkah.")
            else:
                st.info(UNAVAILABLE_SUMMARY.get(guide.status, UNAVAILABLE_SUMMARY["unclear"]))
    show_analysis(guide)


st.set_page_config(page_title="Panduan Pemrograman", layout="wide")
st.html(STYLES)
st.html(
    '<div class="app-masthead"><span class="app-glyph" aria-hidden="true">&gt;_</span>'
    '<span>Panduan Pemrograman</span></div>'
)
if "guide" in st.session_state:
    guide_view(st.session_state.guide)
else:
    upload_view()
