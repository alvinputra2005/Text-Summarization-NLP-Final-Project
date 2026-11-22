import re

from src.segmentation import ACTION, Segment


MAX_STEP_WORDS = 35
SENTENCES = re.compile(r"(?<=[.!?])\s+")
DISFLUENCY = re.compile(r"\b(?:oke|nah|eh|anu|gini|gitu|ntar|kayak|sih|bentar|nih|ya)\b", re.IGNORECASE)
STRETCHED_WORD = re.compile(r"\b[^\W\d_]*(\w)\1{3,}[^\W\d_]*\b", re.IGNORECASE)
FRAGMENT_END = re.compile(r"\b(?:dan|atau|yang|dengan|akan|jika|kalau|menjadi|adalah|ke|dari)\s*[.!?]*$", re.IGNORECASE)


def is_readable_statement(text: str) -> bool:
    words = text.split()
    if not 2 <= len(words) <= MAX_STEP_WORDS or not text.endswith((".", "!")):
        return False
    if len(words) < 4 and not ACTION.search(text):
        return False
    if FRAGMENT_END.search(text) or STRETCHED_WORD.search(text):
        return False
    return len(DISFLUENCY.findall(text)) <= len(words) // 8


def is_readable_summary(text: str) -> bool:
    return len(text.split()) <= MAX_STEP_WORDS and all(map(is_readable_statement, SENTENCES.split(text)))


def readable_coverage(segments: list[Segment]) -> float:
    text = " ".join(segment.text for segment in segments)
    count = len(text.split())
    if not count:
        return 0.0
    return sum(len(sentence.split()) for sentence in SENTENCES.split(text) if is_readable_statement(sentence)) / count
