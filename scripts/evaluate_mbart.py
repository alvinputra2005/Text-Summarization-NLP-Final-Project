"""Evaluate a saved experiment and its base model on the same held-out examples."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import MBart50TokenizerFast

from scripts.train_mbart_pilot import evaluate, load_base_model, sample_split
from src.code_text import technical_terms
from src.evaluation import annotated_detail_recall
from src.summarization import INDONESIAN, PUBLIC_MODEL, details


def literal_recall(predictions):
    recalls = []
    for item in predictions:
        reference = item["reference"]
        prediction = item["prediction"]
        required = technical_terms(reference) | details(reference)[2]
        present = technical_terms(prediction) | details(prediction)[2]
        if required:
            recalls.append(len(required & present) / len(required))
    return mean(recalls) if recalls else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Folder yang berisi metrics.json dan adapter/.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/indosum"))
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--limit", type=int, help="Tanpa opsi ini, evaluasi seluruh split.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Jumlah contoh minimal 1.")
    if args.output.exists():
        parser.error(f"Berkas laporan sudah ada: {args.output}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Jalankan dengan interpreter PyTorch GPU yang mendukung BF16.")

    settings = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint, revision = settings["checkpoint"].split("@", maxsplit=1)
    if checkpoint != PUBLIC_MODEL:
        parser.error(f"Checkpoint eksperimen harus {PUBLIC_MODEL}.")
    rows = sample_split(args.data_dir / f"{args.split}.csv", args.limit)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        checkpoint, revision=revision, src_lang=INDONESIAN, tgt_lang=INDONESIAN, local_files_only=True,
    )
    model = PeftModel.from_pretrained(load_base_model(revision), args.run_dir / "adapter")
    lengths = (settings["source_tokens"], settings["target_tokens"])
    tuned, tuned_predictions = evaluate(model, tokenizer, rows, *lengths, label="Adapter")
    with model.disable_adapter():
        baseline, baseline_predictions = evaluate(model, tokenizer, rows, *lengths, label="Model dasar")
    video_ids = sorted({row["video_id"] for row in rows if row.get("video_id")})
    report = {
        "checkpoint": settings["checkpoint"],
        "adapter": str(args.run_dir / "adapter"),
        "split": args.split,
        "samples": len(rows),
        "videos": video_ids,
        "identical_predictions": sum(
            base["prediction"] == result["prediction"] for base, result in zip(baseline_predictions, tuned_predictions, strict=True)
        ),
        "source_tokens": lengths[0],
        "target_tokens": lengths[1],
        "base": baseline,
        "tuned": tuned,
        "technical_literal_recall": {
            "base": literal_recall(baseline_predictions), "tuned": literal_recall(tuned_predictions),
        },
        "annotated_detail_recall": {
            "base": annotated_detail_recall(rows, baseline_predictions), "tuned": annotated_detail_recall(rows, tuned_predictions),
        },
        "predictions": [
            {**result, "baseline_prediction": base["prediction"], **{
                key: row[key] for key in ("video_id", "start", "end", "topic", "required_details") if row.get(key)
            }}
            for row, result, base in zip(rows, tuned_predictions, baseline_predictions, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Evaluasi tersimpan di {args.output}.")
    print(json.dumps({name: report[name] for name in ("base", "tuned", "identical_predictions", "annotated_detail_recall")}))


if __name__ == "__main__":
    main()
