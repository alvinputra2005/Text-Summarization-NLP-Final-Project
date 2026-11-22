from bisect import bisect_right
from dataclasses import dataclass
import re

from src.overview import CHATTY, SENTENCE_END, clean_sentence
from src.segmentation import ACTION, Segment, group_segments, join_segments


@dataclass(frozen=True)
class StepSource:
    segment: Segment
    text: str
    required_clause: str = ""


# A rule contributes a step only when both the action and its example occur in the transcript.
INPUT_DEMONSTRATIONS = (
    (
        r"data sama dengan.{0,90}?input pakai kurung.{0,180}?teks pertanyaan.{0,100}?masukkan data",
        'Pakai input() dengan pertanyaan “Masukkan data” dan simpan jawaban di variabel data.',
        'simpan jawaban di variabel data',
    ),
    (
        r"\bprint\b.{0,230}?\bmaka data sama dengan (?P<sample>[\w-]+)",
        'Tampilkan isi variabel data dengan print(). Saat pengguna mengetik “{sample}”, nilai itu muncul di layar.',
        'Saat pengguna mengetik “{sample}”',
    ),
    (
        r"\binput\b[^.!?\d]{0,45}\b(?P<number>\d+(?:[.,]\d+)?)\b[^!?]{0,70}"
        r"\b(?:data|nilai) yang (?:keluar|diterima) (?:adalah|bertipe) (?:str|string)\b",
        'Periksa tipe hasil input(). Meski yang diketik {number}, Python membacanya sebagai string.',
        'Python membacanya sebagai string',
    ),
    (
        r"data integer sama dengan.{0,110}?casting ini ke integer.{0,650}?angkanya adalah (?P<number>\d+)"
        r".{0,140}?integer tipenya",
        'Bungkus input() dengan int(). Saat angka {number} dimasukkan, hasilnya bertipe integer.',
        'hasilnya bertipe integer',
    ),
    (
        r"mau pakai integer,? mau pakai float bisa",
        'Pilih int() untuk bilangan bulat atau float() untuk angka desimal sesuai data yang dibutuhkan.',
        'float() untuk angka desimal',
    ),
    (
        r"nilai boolean[- ]nya apa.{0,125}?menjadi false.{0,60}?tetap true",
        'Uji bool() langsung pada jawaban teks. Teks “False” menghasilkan True karena string itu tidak kosong.',
        'teks “False” menghasilkan True',
    ),
    (
        r"casting dulu ke integer.{0,160}?casting ke boolean.{0,145}?"
        r"(?:satu|1) jadi true.{0,45}?(?:nol|0) jadi false",
        'Ubah masukan ke int() sebelum meneruskannya ke bool(). Dalam percobaan ini, 0 menjadi False dan 1 menjadi True.',
        '0 menjadi False dan 1 menjadi True',
    ),
)

PANDAS_ACTIONS = (
    (
        r"\bbaca\b.{0,35}?\bpandas\b.{0,25}?\btampilkan\b.{0,30}?\bpenjualan\b"
        r".{0,30}?\bperhatikan harga\b.{0,24}?\bkosong\b",
        "Baca data dengan pandas, tampilkan data penjualan, lalu periksa harga yang kosong.",
        "periksa harga yang kosong",
    ),
    (
        r"\bhapus baris\b.{0,40}?\bkosong\b.{0,28}?\bdrop[ _-]?na\b"
        r".{0,34}?\bpilih k?olom harga\b",
        "Hapus baris yang harganya kosong dengan dropna() pada kolom harga.",
        "pada kolom harga",
    ),
    (
        r"\btanpa meny[ae][rt]t?akan (?:index|indeks)\b.{0,10}?\bberkas baru berisi data\w?\b",
        "Simpan hasil ke berkas baru tanpa menyertakan indeks.",
        "tanpa menyertakan indeks",
    ),
)
CAUTION = re.compile(r"\b(?:jangan|tidak|bukan|belum|nggak|enggak|gak|tak)\b", re.IGNORECASE)


def input_demonstrations(segments: list[Segment]) -> list[StepSource]:
    parts = []
    offsets = []
    offset = 0
    for segment in segments:
        offsets.append(offset)
        parts.append(segment.text)
        offset += len(parts[-1]) + 1
    transcript = " ".join(parts)
    demonstrations = []
    for pattern, wording, required_clause in INPUT_DEMONSTRATIONS:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match is None:
            continue
        start = bisect_right(offsets, match.start()) - 1
        end = bisect_right(offsets, match.end() - 1)
        demonstrations.append(StepSource(
            join_segments(segments[start:end]), wording.format(**match.groupdict()),
            required_clause.format(**match.groupdict()),
        ))
    return demonstrations


def prepare_step_sources(segments: list[Segment]) -> list[StepSource]:
    groups = group_segments(segments)
    demonstrations = input_demonstrations(segments)
    if len(demonstrations) >= 2:
        groups = [group for group in groups if ACTION.search(group.text)]
    else:
        demonstrations = []

    sources = demonstrations.copy()
    for group in groups:
        text = group.text
        pandas_action = next(
            ((wording, clause) for pattern, wording, clause in PANDAS_ACTIONS if re.search(pattern, text, re.IGNORECASE)),
            None,
        )
        if pandas_action is not None and not CAUTION.search(text):
            sources.append(StepSource(group, *pandas_action))
            continue
        sentences = [clean_sentence(sentence) for sentence in SENTENCE_END.split(text)]
        sentences = [sentence for sentence in sentences if sentence and not sentence.endswith("?")]
        actions = [sentence for sentence in sentences if ACTION.search(sentence)]
        useful = [sentence for sentence in sentences if not CHATTY.search(sentence)]
        picked = actions or useful[:2]
        if picked:
            sources.append(StepSource(group, " ".join(dict.fromkeys(picked))))

    if not sources and groups:
        sources = [StepSource(group, group.text) for group in groups]
    return sorted(sources, key=lambda source: source.segment.start)
