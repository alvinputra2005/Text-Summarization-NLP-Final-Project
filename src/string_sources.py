"""Ground introductory Python string steps in spoken or visible examples."""

import ast
from io import StringIO
import re
import tokenize

from src.segmentation import Segment, join_segments
from src.step_sources import StepSource
from src.transcript_quality import OTHER_PROGRAMMING_LANGUAGE, PYTHON


DEFINITION = re.compile(r"\bstring\b.{0,65}\b(?:kumpulan|kubulan|rangkaian|urutan)\s+(?:dari\s+)?karakter\b", re.I)
SINGLE_QUOTE = re.compile(r"\b(?:single[- ](?:quote|quad)|(?:tanda )?kutip tunggal)\b", re.I)
DOUBLE_QUOTE = re.compile(r"\b(?:double[- ](?:quote|quad)|(?:tanda )?kutip ganda)\b", re.I)
MULTILINE = re.compile(r"\b(?:tiga tanda kutip|triple[- ]quote)\b.{0,65}\b(?:banyak|beberapa|lebih dari satu)\s+baris\b", re.I)
BACKSLASH = re.compile(r"\b(?:tambahi?\w*|tampahin|pakai|tulis)\s+(?:tanda\s+)?backslash\s+(?:lagi\s+)?(?:double|ganda|dua)\b", re.I)
RAW = re.compile(r"\b(?:pakai|gunakan|awali|bikin|buat)\b.{0,42}\braw string\b", re.I)
ESCAPE = re.compile(r"""\\(?:\\|n|t|['"])""")
LITERAL = re.compile(r'''(?i)^([a-z]*)("""|\x27\x27\x27|"|')''')
ESCAPE_WORDS = re.compile(r"\b(?:backslash|escape sequence|raw string)\b", re.I)
STRING_INTRO = re.compile(r"\b(?:pengenalan|cara membuat|cara menulis|deklarasi)\s+string\b", re.I)
QUOTE_WORDS = re.compile(r"\btanda kutip\b", re.I)
TAB_DEMO = re.compile(r"\b(?:kasih|pakai|tambahkan|sisipkan)\s+(?:karakter\s+)?tab\b", re.I)

POINTS = {
    "definition": "String adalah rangkaian karakter.",
    "single": "Buat string dengan mengapit teks menggunakan tanda kutip tunggal `'`.",
    "double": 'Gunakan tanda kutip ganda `"` sebagai pembatas string.',
    "mixed": "Gunakan tanda kutip pembatas yang berbeda dari tanda kutip yang ingin ditampilkan di dalam string.",
    "backslash": "Tulis `\\\\` di dalam string untuk menampilkan satu backslash.",
    "newline": "Sisipkan `\\n` untuk memulai baris baru di dalam string.",
    "tab": "Sisipkan `\\t` untuk menambahkan karakter tab di dalam string.",
    "escape_quote": "Gunakan backslash sebelum tanda kutip yang sama dengan pembatas string agar kutip tersebut menjadi bagian dari teks.",
    "multiline": "Apit teks dengan tiga tanda kutip untuk menulis string beberapa baris.",
    "raw": "Awali literal dengan `r` untuk menulis raw string yang mempertahankan backslash di dalam teks.",
}
POINT_ORDER = ("definition", "quotes", "single", "double", "mixed", "escape_quote",
               "backslash", "newline", "tab", "multiline", "raw")
SPOKEN_RULES = {
    "definition": DEFINITION, "single": SINGLE_QUOTE, "double": DOUBLE_QUOTE, "multiline": MULTILINE,
    "backslash": BACKSLASH, "raw": RAW,
    "newline": re.compile(r"(?i:\bbackslash\s+)n\b|(?<!\\)\\n\b"),
    "tab": re.compile(r"(?i:\bbackslash\s+)t\b|(?<!\\)\\t\b"),
}
ESCAPE_TYPES = {"\\\\": "backslash", "\\n": "newline", "\\t": "tab"}


def is_string_lesson(segments: list[Segment]) -> bool:
    speech = " ".join(cue.text for cue in segments)
    signals = sum(bool(pattern.search(speech)) for pattern in (SINGLE_QUOTE, DOUBLE_QUOTE, ESCAPE_WORDS, MULTILINE))
    focused = signals >= 2 or (
        (DEFINITION.search(speech) or STRING_INTRO.search(speech)) and (signals or QUOTE_WORDS.search(speech))
    )
    return bool(
        not (OTHER_PROGRAMMING_LANGUAGE.search(speech) and not PYTHON.search(speech))
        and len(re.findall(r"\bstring\b", speech, re.I)) >= 2
        and focused
    )


def point(kind: str, source: Segment) -> StepSource:
    text = POINTS[kind]
    return StepSource(source, text, text, allow_source_excerpt=True)


def string_speech_points(segments: list[Segment]) -> dict[str, StepSource]:
    found = {}
    ordered = sorted(segments, key=lambda item: item.start)
    for index, cue in enumerate(ordered):
        after = ordered[index + 1:index + 2]
        paired = [cue, *after] if after and after[0].end > cue.end and after[0].start - cue.end <= 2.5 else [cue]
        for cues in ([cue], paired):
            joined = join_segments(cues)
            for kind, pattern in SPOKEN_RULES.items():
                if kind not in found and pattern.search(joined.text):
                    found[kind] = point(kind, joined)
            if "tab" not in found and TAB_DEMO.search(joined.text):
                text = "Sisipkan karakter tab pada teks untuk memberi jarak."
                found["tab"] = StepSource(joined, text, text, allow_source_excerpt=True)
    return found


def literal_kinds(line: str) -> set[str]:
    """Use Python's syntax and tokens to recognize examples without executing code."""
    if not line or len(line) > 240 or not any(char in line for char in "'\""):
        return set()
    try:
        tree = ast.parse(line.strip())
        if not any(isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call)) for node in ast.walk(tree)):
            return set()
        tokens = list(tokenize.generate_tokens(StringIO(line.strip()).readline))
    except (SyntaxError, ValueError, tokenize.TokenError):
        return set()
    kinds = set()
    for token in tokens:
        if token.type != tokenize.STRING or not (match := LITERAL.match(token.string)):
            continue
        prefix, delimiter = match.groups()
        if prefix.lower() not in {"", "r", "u"}:
            continue
        content = token.string[match.end():-len(delimiter)]
        if not content:
            continue
        if prefix.lower() == "r":
            kinds.add("raw")
            continue
        if len(delimiter) == 3:
            if "\n" in content:
                kinds.add("multiline")
            continue
        kinds.add("single" if delimiter == "'" else "double")
        if ('"' if delimiter == "'" else "'") in content:
            kinds.add("mixed")
        for escape in ESCAPE.findall(content):
            if escape in ESCAPE_TYPES:
                kinds.add(ESCAPE_TYPES[escape])
            elif escape[1] == delimiter:
                kinds.add("escape_quote")
    return kinds


def string_screen_points(segments, parts):
    found = {}
    for part in parts:
        for first, second in zip(part["frames"], part["frames"][1:]):
            start, end = first["second"], second["second"]
            if not 0 < end - start <= 18:
                continue
            common = set(first["lines"]) & set(second["lines"])
            for line in sorted(common):
                for kind in sorted(literal_kinds(line)):
                    if kind not in found:
                        speech = "\n".join(cue.transcript for cue in segments if cue.start < end and cue.end > start)
                        found[kind] = point(kind, Segment(start, end, speech))
    return found


def order_string_points(points):
    # Combining quote styles keeps one instruction when both examples occur in the lesson.
    if {"single", "double"} <= points.keys():
        first, second = sorted((points["single"].segment, points["double"].segment), key=lambda cue: cue.start)
        if second.start - first.end <= 20:
            text = """Apit teks dengan tanda kutip tunggal `'` atau tanda kutip ganda `"`. Pakai pembatas yang sama di awal dan akhir."""
            points = {key: item for key, item in points.items() if key not in {"single", "double"}}
            points["quotes"] = StepSource(join_segments([first, second]), text, text, allow_source_excerpt=True)
    return [source for kind, source in sorted(points.items(), key=lambda item: (
        item[1].segment.start, POINT_ORDER.index(item[0]),
    ))]
