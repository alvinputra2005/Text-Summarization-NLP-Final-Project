import altair as alt
import streamlit as st

from src.evaluation import exact_code_details, normalize, rouge_breakdown, speech_metrics, word_errors


ROUGE_LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rouge3": "ROUGE-3", "rougeL": "ROUGE-L"}


def show_evaluation(transcript: str, summary: str, baseline: str, transcript_only: str = "") -> None:
    with st.form("reference-evaluation", border=False):
        st.caption("Masukkan acuan untuk video ini. Gunakan ringkasan acuan untuk ROUGE dan transkrip acuan untuk WER serta CER.")
        summary_reference = st.text_area("Ringkasan acuan", placeholder="Tulis ringkasan referensi video.")
        comparison = st.text_area("Ringkasan pembanding (opsional)", placeholder="Tempel keluaran model lain untuk dibandingkan dengan acuan yang sama.")
        details = st.text_area("Detail kode acuan (opsional)", placeholder="Satu detail per baris, misalnya angka == 3")
        transcript_reference = st.text_area("Transkrip acuan", placeholder="Tempel transkrip yang sudah diperiksa.")
        calculate = st.form_submit_button("Hitung skor")

    if calculate:
        scores = {}
        candidates = {"Panduan aplikasi": summary}
        if transcript_only:
            candidates["Ringkasan dari transkrip"] = transcript_only
        if normalize(comparison):
            candidates["Ringkasan pembanding"] = comparison
        if normalize(summary_reference):
            scores["rouge"] = {label: rouge_breakdown(summary_reference, text) for label, text in candidates.items()}
            if len(candidates) == 1 and normalize(baseline):
                scores["rouge"]["Teks masukan mBART"] = rouge_breakdown(summary_reference, baseline)
        detail_list = list(dict.fromkeys(line.strip() for line in details.splitlines() if line.strip()))
        if detail_list:
            scores["details"] = {label: exact_code_details(detail_list, [text]) for label, text in candidates.items()}
        if normalize(transcript_reference):
            scores["speech"] = speech_metrics([transcript_reference], [transcript])
            scores["errors"] = word_errors(transcript_reference, transcript)
        st.session_state["guide-evaluation"] = scores

    scores = st.session_state.get("guide-evaluation") or {}
    if not scores:
        st.caption("Skor tersedia setelah teks acuan diisi.")
    if "rouge" in scores:
        show_rouge(scores["rouge"])
    if "details" in scores:
        st.markdown("**Detail kode yang tepat**")
        st.table([{"Keluaran": name, "Cocok": sum(item["present"] for item in hits), "Jumlah acuan": len(hits)}
                  for name, hits in scores["details"].items()])
    if "speech" in scores:
        show_speech(scores["speech"], scores["errors"])


def show_rouge(results: dict[str, dict[str, dict[str, float]]]) -> None:
    st.markdown("**Kesamaan dengan ringkasan acuan**")
    component = st.radio("Komponen ROUGE", ["F1", "Precision", "Recall"], horizontal=True)
    rows = [
        {"Metrik": ROUGE_LABELS[name], "Teks": label, "Skor": parts[component] * 100}
        for label, metrics in results.items() for name, parts in metrics.items()
    ]
    chart = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
        x=alt.X("Metrik:N", sort=list(ROUGE_LABELS.values()), title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Teks:N", sort=list(results)),
        y=alt.Y("Skor:Q", title=f"{component} (0–100)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "Teks:N", scale=alt.Scale(domain=list(results), range=["#2858b8", "#526071", "#9a5c31"]),
            legend=alt.Legend(orient="bottom", columns=1, title=None),
        ),
        tooltip=["Metrik:N", "Teks:N", alt.Tooltip("Skor:Q", format=".2f")],
    ).properties(height=225)
    st.altair_chart(chart)
    st.caption("ROUGE-1, -2, dan -3 memeriksa kecocokan rangkaian kata. ROUGE-L memeriksa kecocokan urutan kata. Arahkan kursor ke batang untuk melihat nilainya.")
    if "Teks masukan mBART" in results:
        st.caption("Teks masukan mBART adalah teks terpilih sebelum peringkasan. Isi kolom pembanding untuk membandingkan keluaran model lain.")


def show_speech(scores: dict[str, float], errors: dict[str, int]) -> None:
    st.markdown("**Kesalahan transkripsi**")
    columns = st.columns(2)
    for column, (name, value) in zip(columns, scores.items(), strict=True):
        column.metric(name.upper(), f"{value:.2%}")
    chart = alt.Chart(alt.Data(values=[{"Jenis": label, "Jumlah": count} for label, count in errors.items()]))
    chart = chart.mark_bar(color="#2858b8", size=13).encode(
        y=alt.Y("Jenis:N", title=None, sort=list(errors)),
        x=alt.X("Jumlah:Q", title="Jumlah kata", axis=alt.Axis(tickMinStep=1)),
        tooltip=["Jenis:N", "Jumlah:Q"],
    ).properties(height=108)
    st.altair_chart(chart)
    st.caption("WER mengukur kesalahan kata dan CER kesalahan karakter. Keduanya bisa melebihi 100% jika banyak sisipan. Diagram merinci operasi penyuntingan kata terhadap acuan.")
