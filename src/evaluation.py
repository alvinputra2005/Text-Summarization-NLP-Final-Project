import re


ROUGE_TYPES = ("rouge1", "rouge2", "rouge3", "rougeL")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def speech_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    import jiwer

    references = [normalize(text) for text in references]
    predictions = [normalize(text) for text in predictions]
    return {"wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions)}


def summary_metrics(reference: str, prediction: str) -> dict[str, float]:
    return {name: parts["F1"] for name, parts in rouge_breakdown(reference, prediction).items()}


def rouge_breakdown(reference: str, prediction: str) -> dict[str, dict[str, float]]:
    from rouge_score import rouge_scorer

    scores = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False).score(reference, prediction)
    return {
        name: {"Precision": score.precision, "Recall": score.recall, "F1": score.fmeasure}
        for name, score in scores.items()
    }


def word_errors(reference: str, prediction: str) -> dict[str, int]:
    import jiwer

    alignment = jiwer.process_words(normalize(reference), normalize(prediction))
    return {"Penggantian": alignment.substitutions, "Penghapusan": alignment.deletions, "Penyisipan": alignment.insertions}
