"""Normalize conversational cues while preserving technical content."""

import re


SPOKEN_FORMS = {
    "ngambil": "mengambil", "masukin": "memasukkan", "jalanin": "menjalankan",
    "enggak": "tidak", "nggak": "tidak", "gak": "tidak",
}
TOKENS = re.compile(
    r'''\x60[^\x60]+\x60|"(?:\\.|[^"\n\\])*"|'(?:\\.|[^'\n\\])*'|\b\w+(?:\.\w+)*\([^()]*\)|'''
    r"\[[^\[\]\n]*\]|\{[^{}\n]*\}|\([^()\n]*\)|"
    r"(?<!\w)[+-]?\d+(?:[.,]\d+)*%?|\w+(?:[.-]\w+)*|>=|<=|!=|==|:=|&&|\|\||[^\w\s]"
)
PROTECTED = re.compile(
    r'''(\x60+[^\x60]*\x60+|"(?:\\.|[^"\n\\])*"|'(?:\\.|[^'\n\\])*'|'''
    r"\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\]|\{[^{}]*\})"
)
ACTION_START = r"kita|coba|gunakan|pakai|jalankan|masukkan|buat|buka|periksa|pilih|ubah|simpan|lanjut(?:kan)?|sekarang|selanjutnya|kemudian|lalu|jadi"
CONTINUATION = ACTION_START + r"|ini|itu|dia|kalau|sebelumnya|terus|misalkan"
AFTER_FILLERS = rf"(?=(?:(?:nah|oke|hmm)(?:\s*,\s*|\s+))*(?:{CONTINUATION})\b)"
LEADING = re.compile(
    r"^(?P<filler>nah|hmm|gitu|gini|nih|deh|dong)\s*,\s*|"
    rf"^(?P<unpunctuated>nah|hmm|oke)(?:\s*,\s*|\s+){AFTER_FILLERS}|"
    rf"^(?P<hesitation>eh|apa\s+ya)\s*,\s*(?=(?:{ACTION_START})\b)",
    re.IGNORECASE,
)
MID_SENTENCE = re.compile(
    rf",\s*(?:(?P<marker>nah|hmm|oke)(?:\s*,\s*|\s+){AFTER_FILLERS}"
    rf"|(?P<hesitation>apa\s+ya)\s*,\s*(?=(?:{ACTION_START})\b))",
    re.IGNORECASE,
)
FILLER_ONLY = re.compile(r"(?:nah|eh|hmm)(?:[\s,]+(?:nah|eh|hmm))*[\s.!…]*", re.IGNORECASE)
IDENTIFIER_PREFIX = re.compile(r"\b(?:variabel|kolom|fungsi|parameter|atribut|method)\s+$", re.IGNORECASE)
IDENTIFIER_SUFFIX = re.compile(r"\s*(?:[=<>]|!=|[(:]|\+=|-=)")
VALUE_PREFIX = re.compile(r"\b(?:status|hasil|jawaban|nilai|label|pesan|teks)\s*$", re.IGNORECASE)


def clean_utterance(source: str) -> tuple[str, list[str]]:
    text = source.strip()
    if FILLER_ONLY.fullmatch(text):
        return "", [word for word in re.findall(r"\w+", text)]

    fillers = []
    while match := LEADING.match(text):
        remaining = text[match.end():]
        if not remaining:
            break
        fillers.append(next(group for group in match.groups() if group is not None))
        text = remaining

    def remove_aside(match):
        filler = match["marker"] or match["hesitation"]
        if filler.casefold() == "oke" and VALUE_PREFIX.search(match.string[:match.start()]):
            return match[0]
        fillers.append(filler)
        return ", "

    chunks = PROTECTED.split(text)
    for index in range(0, len(chunks), 2):
        while (updated := MID_SENTENCE.sub(remove_aside, chunks[index])) != chunks[index]:
            chunks[index] = updated
    text = "".join(chunks)

    def normalize_spoken(match):
        token = match[0]
        if token.casefold() not in SPOKEN_FORMS:
            return token
        if IDENTIFIER_PREFIX.search(text[:match.start()]) or IDENTIFIER_SUFFIX.match(text[match.end():]):
            return token
        return SPOKEN_FORMS[token.casefold()]

    return TOKENS.sub(normalize_spoken, text), fillers
