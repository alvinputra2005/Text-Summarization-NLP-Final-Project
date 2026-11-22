"""Run an mBART LoRA experiment on prepared text and summary pairs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
from statistics import mean

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MBart50TokenizerFast

from src.summarization import INDONESIAN, PUBLIC_MODEL, generate_mbart
from src.evaluation import annotated_detail_recall


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
SEED = 42
SOURCE_TOKENS = 128
TARGET_TOKENS = 48
LEARNING_RATE = 2e-5
ROUGE = ("rouge1", "rouge2", "rougeL")


def load_base_model(revision):
    config = AutoConfig.from_pretrained(PUBLIC_MODEL, revision=revision, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForSeq2SeqLM.from_config(config, attn_implementation="eager")
    checkpoint = hf_hub_download(
        PUBLIC_MODEL, "pytorch_model.bin", revision=revision, local_files_only=True,
    )
    weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    shared_name = "model.shared.weight"
    shared_pointer = weights[shared_name].data_ptr()
    # Cast on CPU one tensor at a time and transfer the tied embedding only once.
    shared_embedding = weights[shared_name].to(DTYPE).to(DEVICE)
    gpu_weights = {
        name: shared_embedding if tensor.data_ptr() == shared_pointer else tensor.to(DTYPE).to(DEVICE)
        for name, tensor in weights.items()
    }
    model.load_state_dict(gpu_weights, assign=True)
    model.tie_weights()
    return model


def sample_split(path, limit):
    with path.open(encoding="utf-8", newline="") as file:
        rows = [
            row for row in csv.DictReader(file) if row["text"].strip() and row["summary"].strip()
        ]
    if limit is None:
        return rows
    if len(rows) < limit:
        raise ValueError(f"{path} hanya berisi {len(rows)} contoh valid, perlu {limit}.")
    return random.Random(SEED).sample(rows, limit)


def encode(tokenizer, row, source_tokens, target_tokens):
    tokenizer.src_lang = row.get("source_lang") or INDONESIAN
    tokenizer.tgt_lang = row.get("target_lang") or INDONESIAN
    inputs = tokenizer(
        row["text"], max_length=source_tokens, truncation=True, return_tensors="pt",
    ).to(DEVICE)
    labels = tokenizer(
        text_target=row["summary"], max_length=target_tokens, truncation=True, return_tensors="pt",
    ).input_ids.to(DEVICE)
    return inputs, labels


def train_epoch(model, tokenizer, rows, source_tokens, target_tokens, optimizer, parameters):
    model.train()
    losses = []
    for step, row in enumerate(rows, start=1):
        inputs, labels = encode(tokenizer, row, source_tokens, target_tokens)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(**inputs, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss tidak finite pada langkah {step}, contoh {row['id']}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
        if step == 1 or step % 10 == 0 or step == len(rows):
            print(f"Train {step}/{len(rows)} | loss {losses[-1]:.4f}", flush=True)
    return mean(losses)


@torch.inference_mode()
def evaluate(model, tokenizer, rows, source_tokens=SOURCE_TOKENS, target_tokens=TARGET_TOKENS, *, label="Validasi"):
    model.eval()
    scorer = rouge_scorer.RougeScorer(ROUGE, use_stemmer=False)
    losses = []
    scores = {name: [] for name in ROUGE}
    predictions = []
    for step, row in enumerate(rows, start=1):
        inputs, labels = encode(tokenizer, row, source_tokens, target_tokens)
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(**inputs, labels=labels, use_cache=False).loss
            generated = generate_mbart(inputs, tokenizer, model, max_new_tokens=target_tokens, target_language=tokenizer.tgt_lang)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss validasi tidak finite untuk contoh {row['id']}.")
        losses.append(loss.item())
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append({"id": row["id"], "reference": row["summary"], "prediction": prediction})
        for name, score in scorer.score(row["summary"], prediction).items():
            scores[name].append(score.fmeasure)
        if step == 1 or step % 5 == 0 or step == len(rows):
            print(f"{label} {step}/{len(rows)} | loss {losses[-1]:.4f}", flush=True)
    return {"validation_loss": mean(losses), **{name: mean(values) for name, values in scores.items()}}, predictions


def evaluate_lead3(rows):
    scorer = rouge_scorer.RougeScorer(ROUGE, use_stemmer=False)
    scores = [
        scorer.score(row["summary"], " ".join(re.split(r"(?<=[.!?])\s+", row["text"], maxsplit=3)[:3]))
        for row in rows
    ]
    return {name: mean(score[name].fmeasure for score in scores) for name in ROUGE}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/indosum"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mbart-pilot"))
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--source-tokens", type=int, default=SOURCE_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--decoder-layers", type=int, default=1, help="Jumlah layer terakhir decoder yang diberi LoRA.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2, help="Jumlah epoch tanpa perbaikan validasi sebelum berhenti.")
    parser.add_argument("--select-by", choices=("rougeL", "details"), default="rougeL",
                        help="Pilih epoch berdasarkan ROUGE-L atau recall detail acuan dengan ROUGE-L sebagai pembanding seri.")
    parser.add_argument("--compare-base", action="store_true", help="Evaluasi model dasar pada contoh validasi yang sama.")
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples) < 1:
        parser.error("Jumlah contoh train dan validation minimal 1.")
    if not 0 < args.learning_rate < 1:
        parser.error("Learning rate harus lebih dari 0 dan kurang dari 1.")
    if not 1 <= args.epochs <= 5 or args.patience < 1:
        parser.error("Gunakan 1 sampai 5 epoch dan patience minimal 1.")
    if args.output_dir.exists():
        parser.error(f"Folder output sudah ada. Pilih folder baru: {args.output_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Jalankan dengan interpreter PyTorch GPU yang mendukung BF16.")

    train_rows = sample_split(args.data_dir / "train.csv", args.train_samples)
    validation_rows = sample_split(args.data_dir / "validation.csv", args.validation_samples)
    empty_predictions = [{"prediction": ""}] * len(validation_rows)
    if args.select_by == "details" and annotated_detail_recall(validation_rows, empty_predictions) is None:
        parser.error("Validasi harus memuat required_details untuk memilih epoch berdasarkan detail acuan.")
    torch.manual_seed(SEED)
    revision = Path(hf_hub_download(PUBLIC_MODEL, "config.json", local_files_only=True)).parent.name
    config = AutoConfig.from_pretrained(PUBLIC_MODEL, revision=revision, local_files_only=True)
    position_limit = config.max_position_embeddings
    if not 3 <= args.source_tokens <= position_limit:
        parser.error(f"Batas masukan harus antara 3 dan {position_limit} token.")
    if not 3 <= args.target_tokens < position_limit:
        parser.error(f"Batas keluaran harus antara 3 dan {position_limit - 1} token.")
    if not 1 <= args.decoder_layers <= config.decoder_layers:
        parser.error(f"Jumlah layer decoder harus antara 1 dan {config.decoder_layers}.")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        PUBLIC_MODEL, revision=revision, src_lang=INDONESIAN, tgt_lang=INDONESIAN, local_files_only=True,
    )
    print(f"{torch.cuda.get_device_name(DEVICE)} | {PUBLIC_MODEL}@{revision}", flush=True)
    print(f"Batas token | masukan {args.source_tokens} | keluaran {args.target_tokens}", flush=True)
    base = load_base_model(revision)
    first_layer = config.decoder_layers - args.decoder_layers
    layer_pattern = "|".join(str(index) for index in range(first_layer, config.decoder_layers))
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            revision=revision,
            target_modules=rf"model\.decoder\.layers\.({layer_pattern})\.(self_attn|encoder_attn)\.(q_proj|v_proj)",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
        ),
    )
    model.print_trainable_parameters()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01, foreach=False)
    args.output_dir.mkdir(parents=True)
    adapter_path = args.output_dir / "adapter"
    lengths = (args.source_tokens, args.target_tokens)
    history = []
    best_epoch = None
    best_score = None
    best_predictions = None
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        if epoch > 1:
            random.Random(SEED + epoch - 1).shuffle(train_rows)
        train_loss = train_epoch(model, tokenizer, train_rows, *lengths, optimizer, parameters)
        metrics, predictions = evaluate(model, tokenizer, validation_rows, *lengths)
        detail = annotated_detail_recall(validation_rows, predictions)
        if detail is not None:
            metrics["annotated_detail_recall"] = detail["recall"]
        result = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(result)
        score = (detail["recall"], metrics["rougeL"]) if args.select_by == "details" else (metrics["rougeL"],)
        if best_score is None or score > best_score:
            best_epoch = result
            best_score = score
            best_predictions = predictions
            model.save_pretrained(adapter_path, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter_path)
            save_json(args.output_dir / "predictions.json", best_predictions)
        save_json(args.output_dir / "epoch_history.json", history)
        print(json.dumps(result), flush=True)
        if epoch - best_epoch["epoch"] >= args.patience:
            print(f"Early stopping: kriteria {args.select_by} pada validasi tidak meningkat.", flush=True)
            break

    model.load_adapter(adapter_path, adapter_name="best")
    model.set_adapter("best")
    model.delete_adapter("default")
    print(f"Adapter epoch {best_epoch['epoch']} berhasil dimuat ulang.", flush=True)
    report = {
        "checkpoint": f"{PUBLIC_MODEL}@{revision}",
        "data_dir": str(args.data_dir),
        "training_sources": dict(sorted(Counter(row.get("source") or "unknown" for row in train_rows).items())),
        "training_languages": dict(sorted(Counter(row.get("target_lang") or INDONESIAN for row in train_rows).items())),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "seed": SEED,
        "epochs": len(history),
        "max_epochs": args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch["epoch"],
        "selection": args.select_by,
        "batch_size": 1,
        "learning_rate": args.learning_rate,
        "decoder_layers": args.decoder_layers,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "source_tokens": args.source_tokens,
        "target_tokens": args.target_tokens,
        **{name: value for name, value in best_epoch.items() if name != "epoch"},
    }
    save_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.compare_base:
        save_json(args.output_dir / "lead3_metrics.json", evaluate_lead3(validation_rows))
        with model.disable_adapter():
            baseline, baseline_predictions = evaluate(
                model, tokenizer, validation_rows, args.source_tokens, args.target_tokens, label="Model dasar",
            )
        baseline["identical_predictions"] = sum(
            trained["prediction"] == untrained["prediction"]
            for trained, untrained in zip(best_predictions, baseline_predictions, strict=True)
        )
        save_json(args.output_dir / "baseline_metrics.json", baseline)
        print(json.dumps(baseline, indent=2), flush=True)


if __name__ == "__main__":
    main()
