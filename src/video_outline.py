"""Select distinct, readable technical statements across a timestamped transcript."""

from bisect import bisect_right
from collections import Counter
from math import isfinite
import re

from src.code_text import technical_terms
from src.evaluation import normalize
from src.overview import LEADING_FILLER, SENTENCE_END, clean_sentence
from src.segmentation import ACTION, Segment, join_segments
from src.step_sources import StepSource, cue_offsets, prepare_step_sources
from src.transcript_quality import (
    DISFLUENCY, FRAGMENT_END, OTHER_PROGRAMMING_LANGUAGE, PROGRAMMING, PYTHON, PYTHON_SYNTAX,
    STRETCHED_WORD, is_readable_summary,
)


BOILERPLATE = re.compile(
    r"\b(?:assalamualaikum|selamat datang|kembali lagi di|terima kasih sudah|sampai jumpa|"
    r"subscribe|channel (?:ini|saya)|jangan lupa (?:like|share|komen)|di video (?:kali ini|selanjutnya))\b", re.I,
)
METADISCOURSE = re.compile(
    r"^(?:(?:sekarang|hari ini|nanti|selanjutnya)\s+)?(?:kita|saya)\s+(?:akan|mau)\s+(?:belajar|bahas)"
    r"|^(?:itu|ini) adalah untuk\b|\b(?:ke sini|ke situ|kayak gini|kayak begitu)\b", re.I,
)
CONCEPT = re.compile(
    r"\b(?:algoritme|aritmatika|array|atribut|argumen|bilangan|boolean|branch|break|casting|class|complex|continue|"
    r"dataframe|decorator|desimal|dictionary|else|elif|encoder|exception|f[ -]?string|float|format|fungsi|"
    r"impor|indeks|integer|iterasi|kompleks|kondisi|konversi|kunci|kurung kurawal|list|loop|metode|modul|objek|operator|"
    r"parameter|pecahan|percabangan|perkalian|persen|persentase|perulangan|pointer|presisi|referensi|return|"
    r"ribuan|serialisasi|string|tipe data|tuple|variabel|variable|"
    r"while|whitespace)\b", re.I,
)
EXPLANATION = re.compile(
    r"\b(?:adalah|agar|akan|artinya|berarti|berfungsi|berisi|bertipe|dipakai|digunakan|dikali|dibagi|"
    r"dimasukkan|ditambah|ditulis|hasil(?:nya)?|kalau|karena|letakkan|memanggil|memasukkan|membandingkan|"
    r"mengakses|menghasilkan|menghapus|menghentikan|mengubah|menyimpan|pakai|pake|periksa|sama dengan|"
    r"terdiri|untuk|gunakan)\b", re.I,
)
RESULT = re.compile(
    r"\b(?:berisi|berhenti|bertipe|dibagi|dihapus|dikali|dikeluarkan|dipanggil|disimpan|ditambah|keluar|"
    r"menampilkan|mengganti|menggabungkan|menghentikan|mencetak|menerima|menjadi|menghasilkan|print|sampai)\b", re.I,
)
REPEATED_DEFINITION = re.compile(r"\b(\w{3,})\s+(?:itu\s+)?(?:adalah|merupakan)\s+\1\b(?=\s*[.,!?]|$)", re.I)
CALCULATION = re.compile(r"\b(\w+)\s+(?:dikali|dibagi|ditambah|dikurangi)\s+(?:dengan\s+)?(\w+)\b", re.I)
GENERIC_TERMS = {"python", "true", "false", "none"}
CONTEXT_DEPENDENT = {"fungsi", "kondisi", "metode", "objek", "tipe data", "variabel"}
STOP_WORDS = frozenset((
    "akan adalah agar aja antara artinya banget berarti bisa buat bukan dari dengan dia disini disitu dulu gini "
    "gitu hasil ini itu jadi kalau kalian kami kapan karena kayak kemudian ketika kita lagi lihat makanya "
    "mana masih mau mereka misalkan misalnya mulai nanti nggak oke pada pakai pake para pasti sama sampai "
    "sangat saja saya sekarang selalu semua setelah sesuatu sini situ sudah supaya teman tanpa tadi telah teman "
    "tentang terus tersebut tetap tidak tuh untuk waktu yang"
).split())
CODE_SHAPE = re.compile(
    r"`[^\s`][^`\n]{0,120}`|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\s*\("
    r"|\b[A-Za-z_]\w*\((?:[\w,'\". +-]{0,80})\)"
    r"|\b[A-Za-z_]\w*\s*(?:==|!=|\+=|:=)\s*[\w'\"-]"
    r"|\b[A-Za-z_]\w*\[\s*(?:-?\d+)?\s*:\s*(?:-?\d+)?\s*(?::\s*-?\d+\s*)?\]"
    r"|\b[A-Za-z_][\w.-]*\.(?:py|js|ts|java|rs|cpp|sql)\b"
)
PROGRAM_CONTEXT = re.compile(
    r"\b(?:kode program|baris kode|blok kode|fungsi bernama|nilai balik|panggilan fungsi|"
    r"struktur data|kompilasi|pemrograman|debugging|source code|source file|kode sumber|deklarasi variabel)\b", re.I,
)
GUIDE_ACTION = re.compile(
    r"\b(?:ambil|aktifkan|buat|cetak|ekspor|hapus|hitung|impor|inisialisasi|isi|jalankan|kembalikan|"
    r"gunakan|kelompokkan|masukkan|muat|panggil|pakai|pasang|pilih|salin|setel|simpan|tambahkan|tulis|ubah|uji|"
    r"taruh|letakkan|membaca|membandingkan|membuat|mengembalikan|mengisi|menjalankan|menjumlahkan|menyalin|menyaring)\b", re.I,
)
UNRESOLVED_START = re.compile(
    r"^(?:(?:jadi|lalu|terus|kemudian)\s+)*(?:ini|itu|dia|di sini|di situ|yang tadi|yang ini)\b", re.I,
)
CONTINUATION = re.compile(r"^(?:agar|supaya|yang|dengan|sehingga|hingga|ke dalam|jika|kecuali)\b", re.I)
OPEN_ACTION = re.compile(r"\b(?:membuat|membaca|menyimpan|menghapus|mengembalikan|menghitung|menampilkan)\W*$", re.I)
LESSON_CHAT = re.compile(
    r"\b(?:kita (?:akan|mau) (?:belajar|bahas|lihat|coba lihat)|gimana kalau kita|gimana caranya|"
    r"kita (?:akan|mau) coba (?:latihan\w*|penggunaan)|topik yang (?:akan )?kita bahas\w*|"
    r"(?:kalian|kamu) bisa (?:download|unduh)|teman-teman semua kembali|kalau dari sebelumnya|"
    r"dari sebelumnya kita sudah|kita sudah bisa)\b", re.I,
)
BACKGROUND_CHAT = re.compile(r"\b(?:pemrograman|tutorial|video) (?:kita )?sebelumnya\b|\bini adalah\b.*\b\w+\.py\b", re.I)
TRAILING_FILLER = re.compile(r"(?:\s+(?:kayak gini|kayak gitu|gitu ya|teman-teman semua|nih|ya))+[.!]?$", re.I)
VAGUE_TAIL = re.compile(r"\b(?:disini|di sini|di situ|disitu|yang ini|yang tadi|dalamnya)\W*$", re.I)
PRINT_WITH_OBJECT = re.compile(r"\bprint\b.{0,28}\b(?:format|nilai|variabel)\s+\w+", re.I)
VAGUE_REPEAT_TAIL = re.compile(r"\b(?:lagi|tadi|adu[h]?|gini|gitu)\W*$", re.I)
WEAK_TERMS = CONTEXT_DEPENDENT | {"format", "print", "not", "or", "and", "input", "type"}
WEAK_RESULT = re.compile(r"^(?:print|sampai)$", re.I)
TEST_RESULT = re.compile(
    r"\b(?:true|false|ribuan|desimal)\b.{0,64}\b(?:jadi|menjadi|adalah)\s+(?:true|false|ribuan|desimal)\b"
    r"|\b(?:koma|titik|coma)\b.{0,22}\b(?:jadi|menjadi|untuk) (?:ribuan|desimal)",
    re.I,
)
VAGUE_RECAP = re.compile(r"\b(?:ini|itu)\s+(?:adalah|merupakan)\s+([\w-]+)\W*$", re.I)
UNRESOLVED_RESULT = re.compile(r"\b(?:menjadi|adalah|merupakan)\s+(?:gini|gitu|ini|itu|yang tadi|yang standarnya)\b", re.I)
DEPENDENT_REFERENCE = re.compile(r"\b(?:dalamnya|ke situ|kolom tadi|fungsi tadi|variabel tadi)\b", re.I)
MAX_GUIDE_POINTS = 10
INSTRUCTION_AFTER_INTRO = re.compile(
    r",\s*(?=(?:(?:lalu|kemudian)\s+)?(?:gunakan|pakai|jalankan|panggil|tulis|tambahkan|"
    r"pilih|simpan|hapus|ubah|buat|letakkan|hitung)\b)", re.I,
)


def has_program_evidence(text: str) -> bool:
    return bool(CODE_SHAPE.search(text) or PROGRAM_CONTEXT.search(text) or PYTHON_SYNTAX.search(text)
                or (PYTHON.search(text) or OTHER_PROGRAMMING_LANGUAGE.search(text)) and PROGRAMMING.search(text))


def local_statement(text: str) -> str:
    text = LEADING_FILLER.sub("", text.strip()).strip(" ,;")
    text = TRAILING_FILLER.sub("", text).rstrip(" ,;")
    if not text or text.endswith("?"):
        return ""
    return text if text.endswith((".", "!")) else text + "."


def is_lesson_chatter(text: str) -> bool:
    return bool(LESSON_CHAT.search(text) or BOILERPLATE.search(text)
                or BACKGROUND_CHAT.search(text) and not GUIDE_ACTION.search(text))


def recoverable_clauses(text):
    for part in SENTENCE_END.split(text):
        if is_lesson_chatter(part):
            fragments = INSTRUCTION_AFTER_INTRO.split(part, maxsplit=1)
            if len(fragments) == 2 and CODE_SHAPE.search(fragments[1]):
                yield fragments[1]
        else:
            yield part


def ranked_source(cues, nearby_program):
    text = local_statement(" ".join(cue.text for cue in cues))
    tokens = normalize(text).split()
    code = has_program_evidence(text) or bool(ACTION.search(text))
    terms = anchors(text) - WEAK_TERMS
    calculation = any(all(token.casefold() not in STOP_WORDS for token in match.groups())
                      for match in CALCULATION.finditer(text))
    terms -= {match[0].casefold() for match in CALCULATION.finditer(text)
              if any(token.casefold() in STOP_WORDS for token in match.groups())}
    result = bool(TEST_RESULT.search(text) or any(not WEAK_RESULT.fullmatch(word) for word in RESULT.findall(text)))
    explicit_action = bool(GUIDE_ACTION.search(text) or (
        PRINT_WITH_OBJECT.search(text) and not VAGUE_REPEAT_TAIL.search(text)))
    action = explicit_action or result or bool(EXPLANATION.search(text))
    fillers = len(DISFLUENCY.findall(text))
    fragment = FRAGMENT_END.search(text) or OPEN_ACTION.search(text)
    recap = VAGUE_RECAP.search(text)
    repeated = recap and recap[1].casefold() in normalize(text[:recap.start()]).split()
    vague_repeat = not code and not result and not explicit_action and (
        VAGUE_REPEAT_TAIL.search(text)
        or len(tokens) <= 10 and any(count > 1 for count in Counter(zip(tokens, tokens[1:])).values())
    )
    if (not 4 <= len(tokens) <= 48 or len(set(tokens)) < 4 or not action
            or not (code or nearby_program and terms or calculation
                    or TEST_RESULT.search(text) or terms and (explicit_action or result))
            or fragment or repeated or vague_repeat or STRETCHED_WORD.search(text)
            or fillers > max(2, len(tokens) // 5)
            or is_lesson_chatter(text) or REPEATED_DEFINITION.search(text)
            or UNRESOLVED_START.search(text) or UNRESOLVED_RESULT.search(text) or VAGUE_TAIL.search(text)):
        return None
    score = 3 * code + min(4, len(terms)) + 2 * explicit_action + 2 * result
    return score - fillers, StepSource(join_segments(cues), text, allow_source_excerpt=True)


def recover_tutorial_sources(segments):
    """Find verifiable statements in a noisy transcript, including cues without punctuation."""
    cues = sorted((cue for cue in segments if cue.text and isfinite(cue.end) and 0 <= cue.start < cue.end),
                  key=lambda cue: cue.start)
    evidence = [has_program_evidence(cue.text) or bool(ACTION.search(cue.text)) for cue in cues]
    if not any(evidence):
        return []

    candidates = []
    for index, cue in enumerate(cues):
        neighbors = range(max(0, index - 2), min(len(cues), index + 3))
        nearby = any(evidence[position] and abs(cues[position].start - cue.start) <= 20
                     for position in neighbors)
        for part in recoverable_clauses(cue.text):
            result = ranked_source([Segment(cue.start, cue.end, part)], nearby)
            if result:
                score, source = result
                candidates.append((score, StepSource(cue, source.text, allow_source_excerpt=True)))
                if index and DEPENDENT_REFERENCE.search(source.text):
                    previous = cues[index - 1]
                    if cue.start - previous.end <= 2 and cue.end - previous.start <= 25 and (
                        anchors(previous.text) or PROGRAMMING.search(previous.text)
                    ):
                        linked = ranked_source([previous, cue], nearby)
                        if linked:
                            candidates.append((linked[0] + 2, linked[1]))
        if index + 1 == len(cues):
            continue
        following = cues[index + 1]
        joins = FRAGMENT_END.search(cue.text) or OPEN_ACTION.search(cue.text) or CONTINUATION.match(following.text)
        if joins and following.start - cue.end <= 2 and following.end - cue.start <= 35:
            result = ranked_source([cue, following], nearby)
            if result:
                candidates.append((result[0] + 1, result[1]))

    duration = max(cue.end for cue in cues) - cues[0].start
    selected = []
    for part in range(4):
        pool = [item for item in candidates
                if min(3, int(4 * (item[1].segment.start - cues[0].start) / duration)) == part]
        selected = add_distinct_sources(selected, [source for _, source in sorted(pool, key=lambda item: -item[0])],
                                        limit=len(selected) + 3)
    supplementary = []
    for source in select_across_video(cues):
        if any(match[0] not in source.segment.text for match in CODE_SHAPE.finditer(source.text)):
            continue
        result = ranked_source([Segment(source.segment.start, source.segment.end, source.text)], False)
        if result:
            supplementary.append(StepSource(source.segment, result[1].text, allow_source_excerpt=True))
    selected = add_distinct_sources(selected, supplementary)
    return sorted(selected, key=lambda source: source.segment.start)


def add_distinct_sources(primary, candidates, *, limit=12):
    selected = [source for source in primary if source.text]
    words = [set(normalize(source.text).split()) for source in selected]
    for source in candidates:
        if len(selected) >= limit:
            break
        tokens = set(normalize(source.text).split())
        if not tokens or any(
            len(tokens & existing) >= 0.65 * len(tokens | existing)
            or (len(tokens & existing) >= 0.36 * min(len(tokens), len(existing))
                and min(source.segment.end, item.segment.end) - max(source.segment.start, item.segment.start)
                >= 0.55 * min(source.segment.end - source.segment.start, item.segment.end - item.segment.start))
            for item, existing in zip(selected, words, strict=True)
        ):
            continue
        selected.append(source)
        words.append(tokens)
    return selected


def prepare_local_outline(segments, adaptive):
    sources = prepare_step_sources(segments)
    if not adaptive:
        return [source if source.required_clause or ranked_source(
                    [Segment(source.segment.start, source.segment.end, source.text)], False)
                else StepSource(source.segment, "") for source in sources]
    verified = [source for source in sources if source.required_clause]
    return sorted(add_distinct_sources(verified, adaptive), key=lambda source: source.segment.start)


def explicit_code_sources(segments):
    return [source for source in recover_tutorial_sources(segments)
            if CODE_SHAPE.search(source.text) and GUIDE_ACTION.search(source.text)]


def limit_tutorial_sources(sources, limit=MAX_GUIDE_POINTS):
    verified = list({source.text: source for source in reversed(sources)
                     if source.text and source.required_clause}.values())[::-1][:limit]
    fallback = [source for source in sources if source.text and not source.required_clause]
    distinct = add_distinct_sources(verified, fallback, limit=limit)
    return sorted(distinct, key=lambda source: source.segment.start) or sources[:1]


def anchors(text):
    operators = {match[0].casefold() for match in CALCULATION.finditer(text)}
    return (technical_terms(text) | operators | {match[0].casefold() for match in CONCEPT.finditer(text)}) - GENERIC_TERMS


def sentence_ranges(text):
    start = 0
    for boundary in SENTENCE_END.finditer(text):
        yield start, boundary.start()
        start = boundary.end()
    if start < len(text):
        yield start, len(text)


def content_phrases(text):
    words = normalize(text).split()
    return {
        f"{first} {second}" for first, second in zip(words, words[1:])
        if len(first) >= 4 and len(second) >= 3 and first not in STOP_WORDS and second not in STOP_WORDS
    }


def select_across_video(segments):
    cues = sorted((cue for cue in segments if cue.text and 0 <= cue.start < cue.end), key=lambda cue: cue.start)
    if not cues:
        return []
    transcript, offsets = cue_offsets(cues)
    candidates = []
    ranges = list(sentence_ranges(transcript))
    spans = ranges + [(first[0], second[1]) for first, second in zip(ranges, ranges[1:])]
    phrase_counts = Counter(phrase for start, end in ranges for phrase in content_phrases(transcript[start:end]))
    for start, end in spans:
        sentence = clean_sentence(transcript[start:end])
        terms = anchors(sentence)
        phrases = {phrase for phrase in content_phrases(sentence) if 2 <= phrase_counts[phrase] <= 10}
        links = EXPLANATION.findall(sentence)
        result = bool(RESULT.search(sentence))
        if (not (terms or phrases) or not (links or result) or not 4 <= len(sentence.split()) <= 30
            or not is_readable_summary(sentence) or sentence.endswith("?")
            or BOILERPLATE.search(sentence) or REPEATED_DEFINITION.search(sentence)
            or (not result and not phrases and terms <= CONTEXT_DEPENDENT)
            or any(METADISCOURSE.search(part) for part in SENTENCE_END.split(sentence))):
            continue
        first = bisect_right(offsets, start) - 1
        stop = bisect_right(offsets, end - 1)
        source = join_segments(cues[first:stop])
        if source.end - source.start <= 40:
            quality = min(len(terms), 4) * 2 + min(len(links), 3) - len(DISFLUENCY.findall(sentence)) * 2
            quality += 3 * result + bool(re.search(r"\b\d+\b", sentence))
            quality += min(3, len(phrases)) * 2
            candidates.append((quality, terms | phrases, set(normalize(sentence).split()), StepSource(source, sentence)))

    earliest, latest = cues[0].start, max(cue.end for cue in cues)
    quarter = (latest - earliest) / 4
    selected = []
    covered = set()
    for number in range(4):
        pool = [row for row in candidates if min(3, int((row[3].segment.start - earliest) / quarter)) == number]
        for _ in range(3):
            usable = [
                row for row in pool if not any(
                    len(row[2] & other[2]) / len(row[2] | other[2]) >= 0.5
                    or (row[3].segment.start, row[3].segment.end) == (other[3].segment.start, other[3].segment.end)
                    for other in selected
                )
            ]
            if not usable:
                break
            best = max(usable, key=lambda row: row[0] + 2 * len(row[1] - covered))
            pool.remove(best)
            selected.append(best)
            covered.update(best[1])
    return [row[3] for row in sorted(selected, key=lambda row: row[3].segment.start)]
