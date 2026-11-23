"""Evaluate a pinned seq2seq model, with optional adapter training."""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter

import torch
from rouge_score import rouge_scorer

from src.evaluation import annotated_detail_recall
from src.seq2seq_models import MODELS, load_runner


SEED = 42
DATASETS = ("indosum", "coding_video_mix")
METRICS = ("rouge1", "rouge2", "rougeL")
SCORER = rouge_scorer.RougeScorer(METRICS, use_stemmer=False)


def experiment_rows(dataset, profile, data_root=Path("data/processed")):
    coding = dataset == "coding_video_mix"
    caps = (12, 6, 9) if profile == "smoke" else ((46, 6, 9) if coding else (100, 20, 30))
    result = {}
    groups = set()
    for split, limit in zip(("train", "validation", "test"), caps, strict=True):
        with (data_root / dataset / f"{split}.csv").open(encoding="utf-8", newline="") as file:
            rows = [
                row for row in csv.DictReader(file)
                if row["text"].strip() and row["summary"].strip()
                and (not coding or (row["source_lang"] == "id_ID" and row["target_lang"] == "id_ID"))
            ]
        identities = {row["id"] for row in rows}
        if len(identities) != len(rows) or len(rows) < limit:
            raise ValueError(f"ID ganda atau jumlah contoh kurang pada {dataset}/{split}.")
        group = {row.get("video_id") or row["id"] for row in rows}
        if groups & group:
            raise ValueError(f"Grup bocor antarsplit pada {dataset}/{split}.")
        groups |= group
        result[split] = random.Random(SEED).sample(rows, limit)
    return result


def score_key(metrics, dataset):
    return (metrics["detail_recall"], metrics["rougeL"]) if dataset == "coding_video_mix" else (metrics["rougeL"],)


def read_comparison(output):
    reports = [
        json.loads((output / dataset / key / "report.json").read_text(encoding="utf-8"))
        for dataset in DATASETS for key in MODELS
    ]
    for dataset in DATASETS:
        hashes = {item["data_hash"] for item in reports if item["dataset"] == dataset}
        if len(hashes) != 1:
            raise ValueError(f"Model pada {dataset} memakai data yang berbeda.")
    return reports


def write_deployment(output, reports, *, base_only=False):
    domain = "coding_video_mix"
    variants = ("base",) if base_only else ("base", "tuned")
    candidates = [(row, variant, row[f"validation_{variant}"])
                  for row in reports if row["dataset"] == domain for variant in variants]
    row, variant, validation = max(
        candidates, key=lambda pair: (*score_key(pair[2], domain), pair[1] == "base", -pair[0]["parameters"]),
    )
    config = {
        "model": row["model"], "hub": row["hub"], "revision": row["revision"], "variant": variant,
        "adapter": f"{domain}/{row['model']}/adapter" if variant == "tuned" else None,
        "source_tokens": row["source_tokens"], "target_tokens": row["target_tokens"],
        "validation": {name: validation[name] for name in ("rougeL", "detail_recall")},
        "profile": row["profile"],
    }
    (output / "deployment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def common_digest(splits):
    data = {split: [[row["id"], row["text"], row["summary"], row.get("required_details", "")]
                    for row in rows] for split, rows in splits.items()}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def encode_rows(runner, rows):
    return [(row, *runner.encode(row["text"], row["summary"])) for row in rows]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(runner, encoded):
    model = runner.model
    model.eval()
    predictions, losses = [], []
    for row, batch, truncated in encoded:
        batch = {key: tensor.to(model.device) for key, tensor in batch.items()}
        loss = float(model(**batch, use_cache=False).loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f"Loss evaluasi tidak finite pada {row['id']}.")
        losses.append(loss)
        synchronize(model.device)
        started = perf_counter()
        text = runner.generate(batch)
        synchronize(model.device)
        elapsed = (perf_counter() - started) * 1_000
        scores = {name: value.fmeasure for name, value in SCORER.score(row["summary"], text).items()}
        details = annotated_detail_recall([row], [{"prediction": text}])
        predictions.append({
            "id": row["id"], "video_id": row.get("video_id", ""), "topic": row.get("topic", ""),
            "reference": row["summary"], "prediction": text, "latency_ms": elapsed,
            "truncated": truncated, **scores, "details": details,
        })
    detail = annotated_detail_recall([row for row, _, _ in encoded], predictions)
    return {
        **{metric: mean(row[metric] for row in predictions) for metric in METRICS},
        "loss": mean(losses),
        "detail_recall": detail["recall"] if detail else None,
        "details": detail,
        "latency_ms": median(item["latency_ms"] for item in predictions),
        "truncated": sum(row["truncated"] for row in predictions),
        "empty": sum(not row["prediction"] for row in predictions),
    }, predictions


def adapter_targets(runner):
    if runner.key == "mt5":
        prefix = f"decoder.block.{len(runner.model.decoder.block) - 1}."
        endings = (".SelfAttention.q", ".SelfAttention.v", ".EncDecAttention.q", ".EncDecAttention.v")
    else:
        prefix = f"model.decoder.layers.{len(runner.model.model.decoder.layers) - 1}."
        endings = (".q_proj", ".v_proj")
    names = [name for name, _ in runner.model.named_modules() if name.startswith(prefix) and name.endswith(endings)]
    if len(names) != 4:
        raise RuntimeError(f"Empat proyeksi decoder tidak ditemukan pada {runner.spec.hub}: {names}")
    return names


def train_epoch(runner, batch_rows, optimizer):
    model = runner.model
    model.train()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    losses = []
    for _, batch, _ in batch_rows:
        optimizer.zero_grad(set_to_none=True)
        inputs = {key: tensor.to(model.device) for key, tensor in batch.items()}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else nullcontext()
        with autocast:
            loss = model(**inputs, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Loss pelatihan tidak finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    return mean(losses)


def compare(args):
    splits = experiment_rows(args.dataset, args.profile)
    source_tokens, target_tokens = (256, 72) if args.dataset == "coding_video_mix" else (512, 96)
    runner = load_runner(args.model, device=args.device, source_tokens=source_tokens, target_tokens=target_tokens,
                         download=args.download)
    print(f"{runner.spec.hub} | {args.dataset} | {args.device}", flush=True)
    report = {
        "dataset": args.dataset, "model": args.model, "hub": runner.spec.hub, "revision": runner.spec.revision,
        "profile": args.profile, "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU",
        "data_hash": common_digest(splits), "counts": {key: len(rows) for key, rows in splits.items()},
        "ids": {key: [row["id"] for row in rows] for key, rows in splits.items()},
        "parameters": runner.model.num_parameters(), "seed": SEED,
        "batch_size": 1, "source_tokens": source_tokens, "target_tokens": target_tokens,
    }
    output = args.output_dir / args.dataset / args.model
    if args.base_only:
        validation, _ = evaluate(runner, encode_rows(runner, splits["validation"]))
        base, predictions = evaluate(runner, encode_rows(runner, splits["test"]))
        report.update(validation_base=validation, base=base, predictions=[{"base": row} for row in predictions],
                      trainable_parameters=0, learning_rate=None, max_epochs=0)
        output.mkdir(parents=True, exist_ok=False)
        write_report(output, report)
        return

    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    targets = adapter_targets(runner)
    runner.model = get_peft_model(runner.model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=targets, revision=runner.spec.revision,
    ))
    n_trainable = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    encoded = {split: encode_rows(runner, rows) for split, rows in splits.items()}
    with runner.model.disable_adapter():
        base_val, _ = evaluate(runner, encoded["validation"])
    starting = {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(runner.model).items()}
    history = []
    winner, best_state = None, None
    grid = ([1e-4] if args.dataset == "coding_video_mix" else [2e-5]) if args.profile == "smoke" else [2e-5, 1e-4]
    epochs = 1 if args.profile == "smoke" else 3
    for lr in grid:
        set_peft_model_state_dict(runner.model, starting)
        params = [parameter for parameter in runner.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, foreach=False)
        best_for_rate = None
        for epoch in range(1, epochs + 1):
            order = random.Random(SEED + epoch).sample(encoded["train"], len(encoded["train"]))
            train_loss = train_epoch(runner, order, optimizer)
            metrics, _ = evaluate(runner, encoded["validation"])
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, **metrics}
            history.append(row)
            print(f"lr={lr:g} epoch={epoch} loss={train_loss:.4f} valid_ROUGE-L={metrics['rougeL']:.4f}", flush=True)
            if winner is None or score_key(row, args.dataset) > score_key(winner, args.dataset):
                winner = row
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in
                              get_peft_model_state_dict(runner.model).items()}
            if best_for_rate is None or score_key(row, args.dataset) > score_key(best_for_rate, args.dataset):
                best_for_rate = row
            if epoch - best_for_rate["epoch"] >= 2:
                break
    set_peft_model_state_dict(runner.model, best_state)
    output.mkdir(parents=True, exist_ok=False)
    runner.model.save_pretrained(output / "adapter", save_embedding_layers=False)
    tuned, tuned_rows = evaluate(runner, encoded["test"])
    with runner.model.disable_adapter():
        base, base_rows = evaluate(runner, encoded["test"])
    report.update({
        "trainable_parameters": n_trainable,
        "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip": 1.0, "lora_rank": 8,
        "lora_alpha": 16, "lora_dropout": 0.05, "max_epochs": epochs, "patience": 2,
        "validation_base": base_val, "validation_tuned": winner, "history": history,
        "base": base, "tuned": tuned,
        "predictions": [{"base": b, "tuned": t} for b, t in zip(base_rows, tuned_rows, strict=True)],
    })
    write_report(output, report)


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--base-only", action="store_true", help="Evaluate validation and test using the base checkpoint.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda:0" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        parser.error("Perangkat CUDA/ROCm BF16 tidak siap.")
    if (args.output_dir / args.dataset / args.model).exists():
        parser.error("Pilih folder keluaran baru untuk eksperimen model ini.")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    compare(args)


if __name__ == "__main__":
    main()
