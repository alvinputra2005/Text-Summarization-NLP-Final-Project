from collections import Counter
from dataclasses import dataclass
import os
import re

import torch
from transformers import AutoModelForSeq2SeqLM, MBart50TokenizerFast

from src.segmentation import ACTION


PUBLIC_MODEL = "facebook/mbart-large-50"
INDONESIAN = "id_ID"
CODE = re.compile(
    r"`[^`]+`|\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\b(?!\s*[.\w])"
    r"|\b[A-Za-z_]\w*(?=\s*\()|\b[A-Za-z_]\w*_[A-Za-z_\d]+\b"
    r"|\b(?:dropna|fillna|isna|isnull|head|describe|print|DataFrame)\b"
    r"|\b(?:subset|axis|inplace|index|sep|delimiter|encoding|dtype|header|skiprows|nrows|True|False|None)\b"
)
ASSIGNMENT = re.compile(r"\b[A-Za-z_]\w*\s*=\s*(?:\[[^\]]*\]|\"[^\"]*\"|'[^']*'|[^\s,)]+)")
NUMBER = re.compile(
    r"(?<!\w)[+\-]?\d+(?:[.,]\d+)*%?(?!\w)"
    r"|\b(?:nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|"
    r"belas|puluh|ratus|ribu|juta|miliar|persen)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:tidak|tak|jangan|bukan|belum|tanpa|nggak|enggak|gak)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Summary:
    text: str
    is_excerpt: bool = False


def model_reference() -> str:
    return os.environ.get("MBART_MODEL_ID", "").strip() or PUBLIC_MODEL


def load_mbart(reference: str):
    tokenizer = MBart50TokenizerFast.from_pretrained(reference, src_lang=INDONESIAN, tgt_lang=INDONESIAN)
    # The public repository publishes PyTorch weights. Load them directly with PyTorch >= 2.6.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        reference, use_safetensors=False if reference == PUBLIC_MODEL else None
    ).to("cpu").eval()
    return tokenizer, model


def generate_mbart(inputs, tokenizer, model, *, max_new_tokens: int = 56):
    language_id = tokenizer.lang_code_to_id[INDONESIAN]
    with torch.inference_mode():
        return model.generate(
            **inputs.to(model.device),
            decoder_start_token_id=tokenizer.eos_token_id,
            forced_bos_token_id=language_id,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            do_sample=False,
            return_dict_in_generate=False,
        )


def summarize(text: str, tokenizer, model) -> Summary:
    inputs = tokenizer(text, return_tensors="pt", verbose=False)
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        return Summary(text, is_excerpt=True)
    output = generate_mbart(inputs, tokenizer, model)
    generated = " ".join(tokenizer.decode(output[0], skip_special_tokens=True).split())
    if summary_preserves_details(text, generated):
        return Summary(generated)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    excerpt = " ".join(sentence for sentence in sentences if ACTION.search(sentence) or any(details(sentence)))
    excerpt = excerpt or sentences[0]
    return Summary(excerpt, is_excerpt=True)


def details(text: str) -> tuple[set[str], set[str], set[str], Counter[str]]:
    return (
        {match.strip("`") for match in CODE.findall(text)},
        {re.sub(r"\s*=\s*", "=", match, count=1) for match in ASSIGNMENT.findall(text)},
        {number.casefold() for number in NUMBER.findall(text)},
        Counter(word.lower() for word in NEGATION.findall(text)),
    )


def summary_preserves_details(source: str, summary: str) -> bool:
    if not summary or len(summary.split()) > min(35, len(source.split())):
        return False
    if details(source) != details(summary):
        return False
    source_actions = {match.lastgroup for match in ACTION.finditer(source)}
    summary_actions = {match.lastgroup for match in ACTION.finditer(summary)}
    if source_actions != summary_actions:
        return False
    if NEGATION.search(source):
        # Pertahankan cakupan kata negasi pada klausa sumber.
        return summary.casefold() in source.casefold()
    source_words = set(re.findall(r"\b\w{4,}\b", source.casefold()))
    summary_words = set(re.findall(r"\b\w{4,}\b", summary.casefold()))
    return len(source_words & summary_words) >= min(2, len(source_words)) > 0
