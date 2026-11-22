from bisect import bisect_right
from dataclasses import dataclass
import re

from src.overview import CHATTY, SENTENCE_END, clean_sentence, join_topics, overview_context, topic_tags
from src.segmentation import ACTION, Segment, group_segments, join_segments
from src.transcript_quality import MAX_STEP_WORDS, OTHER_PROGRAMMING_LANGUAGE, PYTHON, is_readable_statement


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
        r"\bhapus baris\b.{0,20}?\bharga (?:yang )?kosong\b.{0,28}?\bdrop[ _-]?na\b",
        "Gunakan dropna() untuk menghapus baris dengan harga kosong.",
        "menghapus baris dengan harga kosong",
    ),
    (
        r"\btanpa meny(?:[ae][rt]t?akan|atakan) (?:index|indeks)\b.{0,10}?\bberkas baru berisi data\w?\b",
        "Simpan hasil ke berkas baru tanpa menyertakan indeks.",
        "tanpa menyertakan indeks",
    ),
)
CAUTION = re.compile(r"\b(?:jangan|tidak|bukan|belum|nggak|enggak|gak|tak)\b", re.IGNORECASE)
BOOLEAN_CONTEXT = re.compile(r"\b(?:boolean|operasi logika|tabel kebenaran)\b", re.IGNORECASE)
BRANCH_DEMONSTRATIONS = (
    (r"\bif\s+kondisi\s+aksi.{0,180}?\bpakai elif\b.{0,130}?\b(?:lagi kondisi|kondisi lagi)\b",
     "Gunakan if untuk kondisi pertama, lalu elif untuk memeriksa kondisi berikutnya jika kondisi sebelumnya salah."),
    (r"\bpakai elif lagi kondisi lagi\b.{0,150}?\belse\b.{0,140}?\bsebanyak apapun\b",
     "Tambahkan beberapa elif untuk kondisi tambahan. Letakkan else di akhir untuk menangani masukan yang tidak memenuhi kondisi mana pun."),
    (r"\bkhusus untuk si (?P<name>\w+)\b.{0,100}?\bbikin elif\b",
     "Tambahkan elif khusus untuk mengenali nama {name} setelah pemeriksaan nama pertama."),
    (r"\b(?P<name>\w+)\s+(?:nggak|tidak) terdeteksi\b.{0,100}?\belse\b.{0,100}?\b(?:nggak|gak|tidak) kenal\b",
     "Gunakan else untuk nama yang tidak dikenali. Saat menguji nama {name}, program menampilkan pesan bahwa nama itu tidak dikenal."),
)
RANGE_RESULT = re.compile(
    r"\brange\s*\(?\s*(?:(?P<start>-?\d+)\s*,\s*)?(?P<stop>-?\d+)(?!\d|\s*,\s*-?\d)\s*\)?"
    r"(?:(?!\brange\b)[^!?]){0,110}?(?<!\w)(?P<first>-?\d+)\s+sampai\s+(?P<last>-?\d+)\b",
    re.IGNORECASE,
)
INTRO_TOPICS = {
    "input": "masukan pengguna", "string": "teks", "integer": "bilangan bulat", "float": "bilangan desimal",
    "boolean": "nilai Boolean", "file": "berkas data", "pandas": "pengolahan data dengan pandas",
    "branch": "percabangan", "loop": "perulangan",
}
BOOLEAN_EXPERIMENTS = (
    ("not", r"\btrue[ ,]+(?:di[ -]?)?not (?:jadi|menjadi) false\b|\bnot true\s*(?:jadi|menjadi|=)\s*false\b",
     r"\bnot dulu\b|\b(?:uji|coba)\s+(?:operator\s+)?not\b",
     "Uji operator not pada nilai Boolean. Operator ini membalik nilai, sehingga not True menghasilkan False dan not False menghasilkan True."),
    ("or", r"\bfalse or false\s*(?:jadi|menjadi|=)\s*false\b",
     r"\b(?:uji|coba)\s+(?:yang |operator )?or\b",
     "Uji operator or pada dua nilai Boolean. Hasilnya True jika setidaknya satu masukan True, termasuk ketika keduanya True."),
    ("and", r"\bfalse and (?:false|true)\s*(?:jadi|menjadi|=)\s*false\b",
     r"\b(?:uji|coba)\s+(?:yang |operator )?and\b|\bsetelah or itu ada (?:and|n)\b",
     "Uji operator and pada dua nilai Boolean. Hasilnya True hanya ketika kedua masukan True."),
    ("xor", r"\btrue (?:xor|exor|\^) true\s*(?:jadi|menjadi|=)\s*false\b",
     r"\b(?:selanjutnya|uji|coba)\b[^.!?]{0,65}\b(?:xor|exor)\b",
     "Uji XOR dengan operator ^ pada dua nilai Boolean. Hasilnya True jika tepat satu masukan True. Dua masukan True menghasilkan False."),
)


def cue_offsets(segments: list[Segment]) -> tuple[str, list[int]]:
    parts = []
    offsets = []
    offset = 0
    for segment in segments:
        offsets.append(offset)
        parts.append(segment.text)
        offset += len(parts[-1]) + 1
    return " ".join(parts), offsets


def matched_cues(segments: list[Segment], offsets: list[int], match: re.Match) -> Segment:
    start = bisect_right(offsets, match.start()) - 1
    end = bisect_right(offsets, match.end() - 1)
    return join_segments(segments[start:end])


def input_demonstrations(segments: list[Segment], transcript: str, offsets: list[int]) -> list[StepSource]:
    demonstrations = []
    for pattern, wording, required_clause in INPUT_DEMONSTRATIONS:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match is None:
            continue
        demonstrations.append(StepSource(
            matched_cues(segments, offsets, match), wording.format(**match.groupdict()),
            required_clause.format(**match.groupdict()),
        ))
    return demonstrations


def range_demonstrations(segments: list[Segment], transcript: str, offsets: list[int]) -> list[StepSource]:
    demonstrations = []
    seen = set()
    for match in RANGE_RESULT.finditer(transcript):
        start = int(match["start"] or 0)
        stop = int(match["stop"])
        if (start, stop) in seen or not (start <= stop - 1):
            continue
        if (int(match["first"]), int(match["last"])) != (start, stop - 1):
            continue
        call = f"range({start}, {stop})" if match["start"] is not None else f"range({stop})"
        clause = f"menghasilkan angka {start} sampai {stop - 1}"
        text = f"{call} pada perulangan for {clause}. Batas akhir {stop} tidak ikut diambil."
        demonstrations.append(StepSource(matched_cues(segments, offsets, match), text, clause))
        seen.add((start, stop))
    return demonstrations


def boolean_demonstrations(segments: list[Segment]) -> list[StepSource]:
    if not any(BOOLEAN_CONTEXT.search(segment.text) for segment in segments):
        return []
    sources = []
    for _, result, introduction, wording in BOOLEAN_EXPERIMENTS:
        for index, segment in enumerate(segments):
            # Speech recognition can split one expression across two cues.
            cues = segments[index:index + 2]
            if not re.search(result, segment.text, re.IGNORECASE):
                if len(cues) == 1 or re.search(result, cues[1].text, re.IGNORECASE):
                    continue
                if not re.search(result, " ".join(cue.text for cue in cues), re.IGNORECASE):
                    continue
            start = next((
                offset for offset in range(index, -1, -1)
                if segment.start - segments[offset].start <= 175
                and re.search(introduction, segments[offset].text, re.IGNORECASE)
            ), index)
            end = index + (1 if re.search(result, segment.text, re.IGNORECASE) else 2)
            # Preserve the truth condition, including which input combination produces it.
            sources.append(StepSource(join_segments(segments[start:end]), wording, wording.partition(". ")[2]))
            break
    return sources


def prepare_step_sources(segments: list[Segment]) -> list[StepSource]:
    groups = group_segments(segments)
    transcript, offsets = cue_offsets(segments)
    demonstrations = []
    if not OTHER_PROGRAMMING_LANGUAGE.search(transcript) or PYTHON.search(transcript):
        demonstrations = input_demonstrations(segments, transcript, offsets) + boolean_demonstrations(segments)
        demonstrations.extend(range_demonstrations(segments, transcript, offsets))
        for pattern, wording in BRANCH_DEMONSTRATIONS:
            if match := re.search(pattern, transcript, re.IGNORECASE):
                wording = wording.format(**match.groupdict())
                demonstrations.append(StepSource(matched_cues(segments, offsets, match), wording, wording))
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
        sentences = [clean_sentence(sentence) for sentence in SENTENCE_END.split(text)
                     if sentence.endswith((".", "!")) or ACTION.search(sentence)]
        sentences = [sentence for sentence in sentences if is_readable_statement(sentence)]
        actions = [sentence for sentence in sentences if ACTION.search(sentence)]
        useful = [sentence for sentence in sentences if not CHATTY.search(sentence)]
        picked = actions or useful[:2]
        selected = " ".join(dict.fromkeys(picked[:2]))
        if len(selected.split()) > MAX_STEP_WORDS:
            selected = picked[0]
        sources.append(StepSource(group, selected))
    return sorted(sources, key=lambda source: source.segment.start)


def prepare_intro(evidence: tuple[str, ...]) -> str:
    if not evidence:
        return ""
    operators = [name.upper() if name == "xor" else name for name, _, _, wording in BOOLEAN_EXPERIMENTS if wording in evidence]
    if operators and len(operators) == len(evidence):
        return f"Video ini membahas operasi logika Python dengan menguji {join_topics(operators)} pada nilai Boolean."
    tags = topic_tags(" ".join(evidence))
    if {"input", "string", "integer", "float", "boolean"} <= tags:
        return "Video ini membahas cara membaca masukan pengguna dengan input() dan mengubah hasilnya dari string menjadi integer, float, atau boolean."
    sentences = tuple(sentence for item in evidence for sentence in SENTENCE_END.split(item))
    context = SENTENCE_END.split(overview_context(sentences))[0]
    if context.startswith("Video ini membahas ") and context not in sentences and is_readable_statement(context):
        return context
    topics = [description for name, description in INTRO_TOPICS.items() if name in tags]
    return f"Video ini membahas {join_topics(topics)}." if topics else ""
