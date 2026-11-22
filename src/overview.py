from dataclasses import dataclass
import re

from src.segmentation import ACTION, Segment


TOPICS = re.compile(
    r"\b(?P<input>input)\b|\b(?P<string>str|string)\b|\b(?P<integer>int|integer|bilangan bulat)\b|"
    r"\b(?P<float>float|desimal)\b|\b(?P<boolean>bool|boolean|true|false)\b|\b(?P<file>csv|excel|json)\b|"
    r"\b(?P<pandas>pandas|dataframe)\b|\b(?P<branch>if|elif|else|percabangan)\b|\b(?P<loop>perulangan|while|for loop|range)\b",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
LEADING_FILLER = re.compile(r"^(?:(?:nah|oke|eh|jadi|ya)[,.]?\s+)+", re.IGNORECASE)
FILLER = re.compile(r"\b(?:teman-teman semua|teman-teman|gitu|gini|nih|deh|dong)\b", re.IGNORECASE)
EXPLANATION = re.compile(
    r"\b(?:gunakan|pakai|dengan|bertipe|tipe data|selalu|pasti|menghasilkan|"
    r"membaca|mengambil|masukkan|konversi|casting|ubah|menjadi|harus|perlu|sama dengan|baru)\b",
    re.IGNORECASE,
)
CHATTY = re.compile(
    r"\b(?:assalamualaikum|halo semuanya|sampai jumpa|video berikutnya|episode|selamat datang|subscribe|komentar|bye|selanjutnya|gimana|misalkan)\b",
    re.IGNORECASE,
)
DEFINITION = re.compile(r"\b(?:tipe data|bertipe|pasti|selalu|secara bawaan|kalau.*(?:true|false))\b", re.IGNORECASE)
INPUT_EVIDENCE = "Gunakan fungsi input() untuk membaca masukan pengguna."
STRING_EVIDENCE = "Masukan pengguna dibaca sebagai string."
NUMBER_EVIDENCE = "Untuk membaca angka, konversikan masukan ke integer atau float."
BOOLEAN_EVIDENCE = "Untuk memperoleh nilai boolean, konversikan masukan ke integer terlebih dahulu, lalu ke boolean."
PROMPT_EXAMPLE = "Fungsi input() dapat menampilkan teks pertanyaan kepada pengguna."
TEXT_BOOLEAN_EXAMPLE = 'Saat teks "False" langsung dibaca sebagai boolean, hasilnya justru True karena string tersebut tidak kosong.'
BINARY_EXAMPLE = "Dalam contoh masukan numerik, 0 menghasilkan False dan 1 menghasilkan True setelah masukan dikonversi ke integer."
NUMBER_AS_TEXT = re.compile(
    r"\binput\b[^.!?\d]{0,45}\b(\d+(?:[.,]\d+)?)\b[^!?]{0,70}"
    r"\b(?:data|nilai) yang (?:keluar|diterima) (?:adalah|bertipe) (?:str|string)\b",
    re.IGNORECASE,
)
TECHNICAL_REWRITES = (
    (r"\binput\b.*\binput pakai kurung", INPUT_EVIDENCE),
    (r"\bdata yang dimasukkan pasti (?:str|string)\b", STRING_EVIDENCE),
    (r"\bpakai integer\b.*\bpakai float\b", NUMBER_EVIDENCE),
    (r"\bcasting ke integer\b.*\bbaru\b.*\bcasting ke boolean\b", BOOLEAN_EVIDENCE),
)
PROCEDURAL_VERBS = {
    "gunakan": "penggunaan", "pakai": "penggunaan", "baca": "pembacaan", "muat": "pemuatan",
    "simpan": "penyimpanan", "hapus": "penghapusan", "ubah": "perubahan", "periksa": "pemeriksaan",
    "masukkan": "pemasukan", "konversikan": "konversi", "buat": "pembuatan",
    "tampilkan": "penampilan", "jalankan": "eksekusi", "impor": "impor", "ekspor": "ekspor",
    "tambahkan": "penambahan", "pilih": "pemilihan", "klik": "pemilihan",
}
PROCEDURE = re.compile(
    rf"^(?:untuk [^,]+,\s*)?(?P<negation>jangan\s+)?(?P<verb>{'|'.join(PROCEDURAL_VERBS)})\s+(?P<object>.+)$",
    re.IGNORECASE,
)
SEQUENCE = re.compile(r"\b(?:terlebih dahulu|kemudian|setelah itu|berikutnya|lalu|langkah pertama)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Overview:
    text: str
    evidence: tuple[str, ...]
    is_excerpt: bool


def topic_tags(text: str) -> set[str]:
    return {match.lastgroup for pattern in (TOPICS, ACTION) for match in pattern.finditer(text)}


def clean_sentence(sentence: str) -> str:
    sentence = LEADING_FILLER.sub("", sentence.strip())
    sentence = re.sub(r"\s+([,.;!?])", r"\1", " ".join(FILLER.sub("", sentence).split()))
    sentence = re.sub(r"\s*[,;]\s*(?=[.!?])", "", sentence).strip(" ,.;")
    sentence = re.sub(r"\bngambil\b", "mengambil", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bmasukin\b", "memasukkan", sentence, flags=re.IGNORECASE)
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    return sentence if sentence[-1] in "!?" else f"{sentence}."


def find_input_examples(segments: list[Segment]) -> tuple[str, ...]:
    text = " ".join(segment.text for segment in segments)
    examples = []
    if re.search(r"\binput pakai kurung[^.!?]{0,120}[.!?]?\s*Biasanya teks pertanyaan", text, re.IGNORECASE):
        examples.append(PROMPT_EXAMPLE)
    if match := NUMBER_AS_TEXT.search(text):
        examples.append(f"Angka {match[1]} yang diketik tetap terbaca sebagai string.")
    if re.search(r"\b(?:bool|boolean)\b.{0,220}\bfalse\b.{0,60}\btetap true\b", text, re.IGNORECASE):
        examples.append(TEXT_BOOLEAN_EXAMPLE)
    if re.search(r"\b(?:satu|1) jadi true\b.{0,40}\b(?:nol|0) jadi false\b", text, re.IGNORECASE):
        examples.append(BINARY_EXAMPLE)
    return tuple(examples)


def prepare_evidence(segments: list[Segment]) -> tuple[str, ...]:
    pending = ""
    candidates = []
    for segment in segments:
        pieces = SENTENCE_END.split(f"{pending} {segment.text}".strip())
        candidates.extend(pieces[:-1])
        pending = pieces[-1]
        if pending.endswith((".", "?", "!")) or len(pending.split()) >= 28:
            candidates.append(pending)
            pending = ""
    if pending:
        candidates.append(pending)

    ranks = []
    seen = set()
    for index, raw in enumerate(candidates):
        if re.search(r"\b(?:nanti|tutorial selanjutnya)\b.*\b(?:belajar|latihan)\b", raw, re.IGNORECASE):
            continue
        sentence = clean_sentence(raw)
        rewrite = next(
            (value for pattern, value in TECHNICAL_REWRITES if re.search(pattern, sentence, re.IGNORECASE)),
            None,
        )
        if rewrite:
            sentence = rewrite
        tokens = set(re.findall(r"\w+", sentence.casefold()))
        if not tokens or sentence.casefold() in seen:
            continue
        seen.add(sentence.casefold())
        tags = topic_tags(sentence)
        score = 3 * bool(tags) + min(3, len(EXPLANATION.findall(sentence))) * 2
        score += 10 * bool(rewrite)
        score += 5 * bool(DEFINITION.search(sentence) and tags)
        score -= 4 * (len(tags) >= 3 and not EXPLANATION.search(sentence))
        score -= 2 * len(CHATTY.findall(sentence))
        score -= 2 * sentence.endswith("?")
        score -= 3 * (len(tokens) < 5)
        ranks.append((index, sentence, tags, score, tokens))

    selected = []
    covered = set()
    while ranks and len(selected) < 6:
        best = max(ranks, key=lambda item: item[3] + 4 * len(item[2] - covered))
        ranks.remove(best)
        if selected and (best[3] < 1 or (covered and not best[2] - covered)):
            continue
        if any(len(best[4] & other[4]) / len(best[4] | other[4]) > 0.55 for other in selected):
            continue
        selected.append(best)
        covered.update(best[2])
    evidence = tuple(item[1] for item in sorted(selected))
    if {INPUT_EVIDENCE, STRING_EVIDENCE} <= set(evidence):
        return evidence + find_input_examples(segments)
    return evidence


def join_topics(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " dan " + items[-1]


def overview_context(evidence: tuple[str, ...]) -> str:
    context = []
    facts = list(evidence)
    # Combine recurring explanations of Python input into one conceptual account.
    if {INPUT_EVIDENCE, STRING_EVIDENCE} <= set(evidence):
        opening = (
            "Pemateri memperagakan cara input() menampilkan pertanyaan dan membaca jawaban pengguna."
            if PROMPT_EXAMPLE in evidence
            else "Video ini membahas cara Python membaca masukan pengguna dengan input()."
        )
        context.append(opening)
        examples = [sentence for sentence in evidence if is_input_example(sentence)]
        numeric_text = next((example for example in examples if example.startswith("Angka ")), None)
        conversion = []
        if NUMBER_EVIDENCE in evidence:
            conversion.append("nilai numerik seperti integer atau float")
        if BOOLEAN_EVIDENCE in evidence and not {TEXT_BOOLEAN_EXAMPLE, BINARY_EXAMPLE}.intersection(examples):
            conversion.append("nilai boolean")
        if numeric_text:
            context.append(numeric_text)
            if conversion:
                context.append(f"Perbedaan antara teks dan angka ini menjelaskan mengapa masukan perlu dikonversi untuk memperoleh {join_topics(conversion)}.")
        elif conversion:
            context.append(f"Masukan berupa string secara bawaan, sehingga konversi tipe diperlukan untuk memperoleh {join_topics(conversion)}.")
        else:
            context.append("Masukan yang diterima berupa string.")
        context.extend(example for example in (TEXT_BOOLEAN_EXAMPLE, BINARY_EXAMPLE) if example in examples)
        input_facts = {INPUT_EVIDENCE, STRING_EVIDENCE, NUMBER_EVIDENCE, BOOLEAN_EVIDENCE, *examples}
        facts = [sentence for sentence in facts if sentence not in input_facts]

    topics = []
    statements = []
    for sentence in facts:
        sentence = re.sub(r"^(?:kemudian|setelah itu|berikutnya|lalu)[, ]+\s*", "", sentence, flags=re.IGNORECASE)
        match = PROCEDURE.fullmatch(sentence)
        if match:
            subject = SEQUENCE.split(match["object"], maxsplit=1)[0].rstrip(" .,;!?")
            if subject:
                caution = "larangan " if match["negation"] else ""
                topics.append(f"{caution}{PROCEDURAL_VERBS[match['verb'].lower()]} {subject}")
        elif not sentence.endswith("?"):
            statements.append(sentence)
    if topics:
        context.append(f"{'Video ini membahas' if not context else 'Pembahasan juga mencakup'} {join_topics(topics)}.")
    context.extend(statements)
    return " ".join(context) or f"Video ini membahas {evidence[0].rstrip('?! .')}."


def is_input_example(sentence: str) -> bool:
    return sentence in {PROMPT_EXAMPLE, TEXT_BOOLEAN_EXAMPLE, BINARY_EXAMPLE} or bool(
        re.fullmatch(r"Angka \d+(?:[.,]\d+)? yang diketik tetap terbaca sebagai string\.", sentence)
    )


def format_overview(text: str) -> str:
    for example in (TEXT_BOOLEAN_EXAMPLE, BINARY_EXAMPLE):
        lead, marker, rest = text.partition(example)
        if marker and lead.strip():
            return f"{lead.rstrip()}\n\n{marker}{rest}"
    return text


def summarize_overview(evidence: tuple[str, ...], tokenizer, model) -> Overview:
    from src.summarization import details, generate_mbart

    source = overview_context(evidence)
    inputs = tokenizer(source, return_tensors="pt", verbose=False)
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        return Overview(format_overview(source), evidence, True)
    output = generate_mbart(inputs, tokenizer, model, max_new_tokens=180)
    generated = " ".join(tokenizer.decode(output[0], skip_special_tokens=True).split())
    if not generated or generated[-1] not in ".!?":
        generated = SENTENCE_END.split(generated)[:-1]
        generated = " ".join(generated)
    source_tags = topic_tags(source)
    generated_tags = topic_tags(generated)
    normalized_sentences = [sentence.casefold() for sentence in SENTENCE_END.split(generated)]
    source_details = details(source)
    generated_details = details(generated)
    required_examples = [
        item for item in evidence if item != PROMPT_EXAMPLE and is_input_example(item) and item in source
    ]
    if (
        generated and source_tags == generated_tags
        and all(candidate <= original for candidate, original in zip(generated_details, source_details))
        and source_details[-1] == generated_details[-1]
        and all(example.casefold() in generated.casefold() for example in required_examples)
        and len(normalized_sentences) == len(set(normalized_sentences))
        and not any(PROCEDURE.fullmatch(sentence) for sentence in normalized_sentences)
        and not SEQUENCE.search(generated)
        and not re.search(r"\b(\w+)(?:[ ,]+\1){2,}\b", generated, re.IGNORECASE)
        and len(generated.split()) <= len(source.split())
    ):
        return Overview(format_overview(generated), evidence, False)
    return Overview(format_overview(source), evidence, True)
