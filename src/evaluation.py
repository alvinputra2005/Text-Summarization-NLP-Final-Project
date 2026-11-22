import re


ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def speech_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    import jiwer

    references = [normalize(text) for text in references]
    predictions = [normalize(text) for text in predictions]
    return {"wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions)}


def summary_metrics(reference: str, prediction: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scores = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False).score(reference, prediction)
    return {name: score.fmeasure for name, score in scores.items()}
