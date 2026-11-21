"""Run a one-epoch mBART LoRA pilot on the prepared IndoSum splits."""

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import mean

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MBart50TokenizerFast

from src.summarization import INDONESIAN, PUBLIC_MODEL, generate_mbart


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
    if len(rows) < limit:
        raise ValueError(f"{path} hanya berisi {len(rows)} contoh valid, perlu {limit}.")
    return random.Random(SEED).sample(rows, limit)


def encode(tokenizer, row, source_tokens, target_tokens):
    inputs = tokenizer(
        row["text"], max_length=source_tokens, truncation=True, return_tensors="pt",
    ).to(DEVICE)
    labels = tokenizer(
        text_target=row["summary"], max_length=target_tokens, truncation=True, return_tensors="pt",
    ).input_ids.to(DEVICE)
    return inputs, labels


def train_epoch(model, tokenizer, rows, source_tokens, target_tokens):
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01, foreach=False)
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
            generated = generate_mbart(inputs, tokenizer, model, max_new_tokens=target_tokens)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/indosum"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mbart-pilot"))
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--source-tokens", type=int, default=SOURCE_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--compare-base", action="store_true", help="Evaluasi model dasar pada contoh validasi yang sama.")
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples) < 1:
        parser.error("Jumlah contoh train dan validation minimal 1.")
    if args.output_dir.exists():
        parser.error(f"Folder output sudah ada. Pilih folder baru: {args.output_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Jalankan dengan interpreter PyTorch GPU yang mendukung BF16.")

    train_rows = sample_split(args.data_dir / "train.csv", args.train_samples)
    validation_rows = sample_split(args.data_dir / "validation.csv", args.validation_samples)
    torch.manual_seed(SEED)
    revision = Path(hf_hub_download(PUBLIC_MODEL, "config.json", local_files_only=True)).parent.name
    config = AutoConfig.from_pretrained(PUBLIC_MODEL, revision=revision, local_files_only=True)
    position_limit = config.max_position_embeddings
    if not 3 <= args.source_tokens <= position_limit:
        parser.error(f"Batas masukan harus antara 3 dan {position_limit} token.")
    if not 3 <= args.target_tokens < position_limit:
        parser.error(f"Batas keluaran harus antara 3 dan {position_limit - 1} token.")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        PUBLIC_MODEL, revision=revision, src_lang=INDONESIAN, tgt_lang=INDONESIAN, local_files_only=True,
    )
    print(f"{torch.cuda.get_device_name(DEVICE)} | {PUBLIC_MODEL}@{revision}", flush=True)
    print(f"Batas token | masukan {args.source_tokens} | keluaran {args.target_tokens}", flush=True)
    base = load_base_model(revision)
    last_layer = base.config.decoder_layers - 1
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            revision=revision,
            target_modules=rf"model\.decoder\.layers\.{last_layer}\.(self_attn|encoder_attn)\.(q_proj|v_proj)",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
        ),
    )
    model.print_trainable_parameters()
    train_loss = train_epoch(model, tokenizer, train_rows, args.source_tokens, args.target_tokens)
    adapter_path = args.output_dir / "adapter"
    model.save_pretrained(adapter_path, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_path)
    model.load_adapter(adapter_path, adapter_name="reloaded")
    model.set_adapter("reloaded")
    model.delete_adapter("default")
    print(f"Adapter berhasil dimuat ulang dari {adapter_path}. Validasi berjalan.", flush=True)
    metrics, predictions = evaluate(model, tokenizer, validation_rows, args.source_tokens, args.target_tokens)
    report = {
        "checkpoint": f"{PUBLIC_MODEL}@{revision}",
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "seed": SEED,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "source_tokens": args.source_tokens,
        "target_tokens": args.target_tokens,
        "train_loss": train_loss,
        **metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.compare_base:
        with model.disable_adapter():
            baseline, baseline_predictions = evaluate(
                model, tokenizer, validation_rows, args.source_tokens, args.target_tokens, label="Model dasar",
            )
        baseline["identical_predictions"] = sum(
            trained["prediction"] == untrained["prediction"]
            for trained, untrained in zip(predictions, baseline_predictions, strict=True)
        )
        (args.output_dir / "baseline_metrics.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(baseline, indent=2), flush=True)


if __name__ == "__main__":
    main()
