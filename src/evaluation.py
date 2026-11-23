import json
import re


ROUGE_TYPES = ("rouge1", "rouge2", "rouge3", "rougeL")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def exact_code_details(details: list[str], points: list[str]) -> list[dict]:
    results = []
    for detail in details:
        tokens = re.findall(r"\w+|==|\+=|[^\s]", detail)
        pattern = r"\s*".join(map(re.escape, tokens))
        present = bool(tokens) and any(re.search(rf"(?<!\w){pattern}(?!\w|\.\d)", point) for point in points)
        results.append({"detail": detail, "present": present})
    return results


def annotated_detail_recall(rows, predictions):
    totals = 0
    matched = 0
    for row, item in zip(rows, predictions, strict=True):
        if not row.get("required_details"):
            continue
        prediction = normalize(item["prediction"])
        for detail in json.loads(row["required_details"]):
            normalized = normalize(detail)
            if normalized:
                totals += 1
                matched += bool(re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", prediction))
    return {"matched": matched, "total": totals, "recall": matched / totals} if totals else None


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
