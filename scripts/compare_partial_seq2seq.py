"""Compare three seq2seq models trained directly on each dataset."""

import argparse
import json
from pathlib import Path

import torch

from scripts.compare_seq2seq import DATASETS, common_digest, encode_rows, evaluate, experiment_rows
from scripts.notebook_training import SEED, fit_mbart, mbart_parameters, restore, save_report, snapshot
from src.seq2seq_models import MODELS, load_runner


MODELS_IN_COMPARISON = ("mbart", "mt5", "indobart-v2")
PROTOCOL = "supervised-summary-v2"
ARCHIVED_PROTOCOL = "partial-cross-attention-v1"
BENCHMARK_MODELS = {
    "indosum": MODELS_IN_COMPARISON,
    "coding_video_mix": MODELS_IN_COMPARISON,
    "multilingual_code": ("mbart", "mt5"),
}


def comparison_reports(folder, *, datasets=DATASETS):
    reports = []
    for dataset in datasets:
        rows = []
        for model in BENCHMARK_MODELS[dataset]:
            path = folder / dataset / model / "report.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row["dataset"], row["model"], row["revision"]) != (dataset, model, MODELS[model].revision) or (
                row["protocol"] not in {PROTOCOL, ARCHIVED_PROTOCOL}
            ):
                raise ValueError(f"Laporan tidak cocok dengan eksperimen ini: {path}")
            rows.append(row)
        comparable = {(row["data_hash"], row["source_tokens"], row["target_tokens"], row["epochs_limit"],
                       row["batch_size"], row["accumulation"], row["profile"], row["protocol"],
                       row.get("patience", 2)) for row in rows}
        if len(comparable) != 1:
            raise ValueError(f"Model pada {dataset} memakai protokol berbeda.")
        reports.extend(rows)
    return reports


def partial_parameters(runner):
    if runner.key == "mt5":
        from peft import LoraConfig, TaskType, get_peft_model

        runner.model = get_peft_model(runner.model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
            target_modules=["q", "v", "lm_head"],
        ))
        return {name: parameter for name, parameter in runner.model.named_parameters() if parameter.requires_grad}
    params = mbart_parameters(runner.model)
    if len(params) != 2:
        raise RuntimeError(f"Dua proyeksi cross-attention tidak ditemukan untuk {runner.key}.")
    return params


def score_validation(metrics):
    return metrics["rougeL"], metrics["detail_recall"] or 0


def compare_model(key, output, profile, epochs, device, *, download=False, accumulation=None, patience=2,
                  indosum_limits=None, datasets=DATASETS):
    dtype = torch.bfloat16
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float32
    print(f"Menyiapkan {key} pada {device}. Pengunduhan checkpoint pertama dapat memakan waktu.", flush=True)
    runner = load_runner(key, device=device, dtype=dtype, download=download)
    runner.model.requires_grad_(False)
    torch.manual_seed(SEED)
    params = partial_parameters(runner)
    method = "LoRA Q/V dan kepala keluaran" if key == "mt5" else "Q/V cross-attention decoder terakhir"
    learning_rate = 3e-4 if key == "mt5" else 2e-5
    original = snapshot(params)
    for parameter in params.values():
        parameter.requires_grad_(True)
    if accumulation is None:
        accumulation = 1 if profile == "smoke" else 8
    for dataset in datasets:
        runner.set_summary_language("en_XX" if dataset == "multilingual_code" else "id_ID")
        folder = output / dataset / key
        folder.mkdir(parents=True, exist_ok=False)
        limits = indosum_limits if dataset == "indosum" else None
        splits = experiment_rows(dataset, profile, limits=limits)
        if profile == "smoke":
            splits = {role: rows if role == "test" else rows[:2] for role, rows in splits.items()}
        runner.source_tokens, runner.target_tokens = (512, 96) if dataset == "indosum" else (256, 72)
        batches = {role: encode_rows(runner, rows) for role, rows in splits.items()}
        restore(params, original)
        torch.manual_seed(SEED)
        baseline_validation = None
        if key == "mbart" and dataset == "coding_video_mix":
            print(f"{key} {dataset} · evaluasi validasi sebelum latihan", flush=True)
            baseline_validation, _ = evaluate(runner, batches["validation"])
        stage = fit_mbart(runner, params, batches["train"], batches["validation"], dataset, epochs,
                          accumulation=accumulation, patience=patience, selection=score_validation,
                          learning_rate=learning_rate)
        if key == "mt5":
            checkpoint = folder / "adapter"
            runner.model.save_pretrained(checkpoint, save_embedding_layers=False)
        else:
            checkpoint = folder / "selected.pt"
            torch.save({"model": key, "base": runner.spec.hub, "revision": runner.spec.revision,
                        "params": snapshot(params), "source_tokens": runner.source_tokens,
                        "target_tokens": runner.target_tokens, "mode": profile, "epoch": stage["best_epoch"],
                        "strategy": f"langsung-{dataset}"}, checkpoint)
        print(f"{key} {dataset} · menguji epoch {stage['best_epoch']} pada {len(batches['test'])} contoh", flush=True)
        tuned, tuned_predictions = evaluate(runner, batches["test"])
        restore(params, original)
        print(f"{key} {dataset} · menguji checkpoint sebelum latihan pada contoh yang sama", flush=True)
        base, base_predictions = evaluate(runner, batches["test"])
        if baseline_validation is not None:
            tuned_validation = stage["history"][stage["best_epoch"] - 1]["validation"]
            if (tuned_validation["rougeL"] > baseline_validation["rougeL"]
                    and tuned_validation["detail_recall"] is not None
                    and baseline_validation["detail_recall"] is not None
                    and tuned_validation["detail_recall"] >= baseline_validation["detail_recall"]):
                save_report(folder / "deployment.json", {
                    "model": key, "revision": runner.spec.revision, "adapter": None,
                    "partial_weights": checkpoint.name, "source_tokens": runner.source_tokens,
                    "target_tokens": runner.target_tokens,
                })
        save_report(folder / "report.json", {
            "model": key, "dataset": dataset, "hub": runner.spec.hub, "revision": runner.spec.revision,
            "protocol": PROTOCOL, "profile": profile, "device": device, "data_hash": common_digest(splits),
            "counts": {role: len(rows) for role, rows in splits.items()}, "epochs_limit": epochs,
            "source_tokens": runner.source_tokens, "target_tokens": runner.target_tokens,
            "learning_rate": learning_rate, "method": method,
            "batch_size": 1, "accumulation": accumulation, "patience": patience,
            "trainable_parameters": sum(p.numel() for p in params.values()),
            "best_epoch": stage["best_epoch"], "history": stage["history"], "checkpoint": str(checkpoint),
            "base": base, "test": tuned, "base_predictions": base_predictions, "test_predictions": tuned_predictions,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS_IN_COMPARISON, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--patience", type=int, default=2, help="0 untuk menjalankan semua epoch.")
    parser.add_argument("--indosum-limits", type=int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_MODELS), action="append")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if ((args.epochs is not None and args.epochs < 1)
            or (args.accumulation is not None and args.accumulation < 1) or args.patience < 0
            or (args.indosum_limits is not None and min(args.indosum_limits) < 1)):
        parser.error("Epoch dan akumulasi minimal 1. Patience minimal 0.")
    datasets = tuple(dict.fromkeys(args.benchmark)) if args.benchmark else DATASETS
    if any(args.model not in BENCHMARK_MODELS[dataset] for dataset in datasets):
        parser.error(f"{args.model} tidak mendukung bahasa ringkasan pada benchmark yang dipilih.")
    torch.set_num_threads(2)
    compare_model(args.model, args.output, args.profile, args.epochs or (2 if args.profile == "smoke" else 5),
                  args.device, download=args.download, accumulation=args.accumulation, patience=args.patience,
                  indosum_limits=args.indosum_limits, datasets=datasets)


if __name__ == "__main__":
    main()
