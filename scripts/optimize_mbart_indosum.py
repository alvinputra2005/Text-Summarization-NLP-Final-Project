"""Tune mBART on IndoSum validation, then evaluate the selected setup on the fixed test subset."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from scripts.compare_seq2seq import common_digest, encode_rows, evaluate, experiment_rows
from scripts.notebook_training import SEED, fit_mbart, load_mbart_weights, mbart_parameters, restore, save_report, snapshot
from src.seq2seq_models import MODELS, load_runner


GENERATION_GRID = ((96, 2), (160, 2), (160, 5), (192, 5))
TRAINING_TARGET_TOKENS = 160


def validation_candidates(runner, validation, method):
    candidates = []
    for tokens, beams in GENERATION_GRID:
        runner.generation_tokens, runner.generation_beams = tokens, beams
        score, _ = evaluate(runner, validation)
        candidates.append({"method": method, "tokens": tokens, "beams": beams, "validation": score})
        print(f"{method} · {tokens} token · {beams} beam · ROUGE-L validasi {score['rougeL']:.4f}", flush=True)
    return candidates


def choose_candidate(candidates):
    return max(candidates, key=lambda row: (row["validation"]["rougeL"], row["validation"]["rouge2"],
                                            -row["beams"], -row["tokens"]))


def previous_checkpoint(path, valid_count, test_count):
    report = json.loads(path.read_text(encoding="utf-8"))
    counts = report["counts"]
    if (report["dataset"], report["model"], report["revision"], report["profile"],
        counts["validation"], counts["test"]) != (
        "indosum", "mbart", MODELS["mbart"].revision, "pilot", valid_count, test_count
    ):
        raise ValueError("Laporan mBART bagian 13 harus memakai revisi model serta jumlah validasi dan uji yang sama.")
    prior_splits = experiment_rows("indosum", "pilot", limits=(counts["train"], valid_count, test_count))
    if report["data_hash"] != common_digest(prior_splits):
        raise ValueError("Laporan mBART bagian 13 memakai data atau urutan sampel yang berbeda.")
    checkpoint = path.with_name("selected.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Simpan report.json dan selected.pt dari bagian 13 dalam folder yang sama: {path.parent}")
    return checkpoint, report["best_epoch"], counts["train"]


def add_decoder_lora(runner):
    from peft import LoraConfig, TaskType, get_peft_model

    layers = runner.model.model.decoder.layers
    targets = [
        name for name, _ in runner.model.named_modules()
        if any(name.startswith(f"model.decoder.layers.{index}.") for index in range(max(0, len(layers) - 4), len(layers)))
        and name.endswith((".q_proj", ".v_proj"))
    ]
    if not targets:
        raise RuntimeError("Proyeksi Q/V decoder mBART tidak ditemukan untuk LoRA.")
    runner.model = get_peft_model(runner.model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=targets, revision=runner.spec.revision,
    ))
    return {name: parameter for name, parameter in runner.model.named_parameters() if parameter.requires_grad}


def optimize(output, device, counts, epochs, accumulation, patience, *, prior_report=None, train_lora=True):
    train_count, valid_count, test_count = counts
    splits = experiment_rows("indosum", "pilot", limits=counts)
    prior = previous_checkpoint(prior_report, valid_count, test_count) if prior_report else None
    dtype = torch.bfloat16
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float32
    print(f"Memuat mBART Large pada {device}. Pemilihan konfigurasi memakai {valid_count} artikel validasi.", flush=True)
    runner = load_runner("mbart", device=device, dtype=dtype, download=True, source_tokens=512,
                         target_tokens=TRAINING_TARGET_TOKENS)
    validation = encode_rows(runner, splits["validation"])
    original = snapshot(mbart_parameters(runner.model))
    candidates = validation_candidates(runner, validation, "Pralatih")
    training = {}
    if prior:
        checkpoint, prior_epoch, prior_train_count = prior
        load_mbart_weights(runner, checkpoint)
        runner.source_tokens, runner.target_tokens = 512, TRAINING_TARGET_TOKENS
        candidates += validation_candidates(runner, validation, "Fine-tuning bagian 13")
        training["Fine-tuning bagian 13"] = {"best_epoch": prior_epoch, "train_count": prior_train_count}
        restore(mbart_parameters(runner.model), original)

    if train_lora:
        torch.manual_seed(SEED)
        params = add_decoder_lora(runner)
        runner.generation_tokens, runner.generation_beams = 160, 2
        stage = fit_mbart(runner, params, encode_rows(runner, splits["train"]), validation, "indosum",
                          epochs, accumulation=accumulation, patience=patience, learning_rate=1e-4)
        training["LoRA decoder"] = {**stage, "train_count": train_count, "learning_rate": 1e-4,
                                    "parameters": sum(parameter.numel() for parameter in params.values())}
        candidates += validation_candidates(runner, validation, "LoRA decoder")

    winner = choose_candidate(candidates)
    output.mkdir(parents=True, exist_ok=True)
    artifact = None
    if train_lora and winner["method"] == "LoRA decoder":
        artifact = output / "adapter"
        runner.model.save_pretrained(artifact, save_embedding_layers=False)
    elif train_lora:
        runner.model = runner.model.unload()
    if winner["method"] == "Fine-tuning bagian 13":
        checkpoint = prior[0]
        load_mbart_weights(runner, checkpoint)
        artifact = output / "selected.pt"
        shutil.copy2(checkpoint, artifact)
    runner.source_tokens, runner.target_tokens = 512, TRAINING_TARGET_TOKENS
    test = encode_rows(runner, splits["test"])
    runner.generation_tokens, runner.generation_beams = winner["tokens"], winner["beams"]
    print(f"Setelan terpilih: {winner['method']} · evaluasi pada {test_count} artikel uji.", flush=True)
    selected_test, selected_predictions = evaluate(runner, test)
    original_candidate = candidates[0]
    if winner is original_candidate:
        baseline_test, baseline_predictions = selected_test, selected_predictions
    else:
        if winner["method"] == "LoRA decoder":
            runner.model = runner.model.unload()
        restore(mbart_parameters(runner.model), original)
        runner.generation_tokens, runner.generation_beams = original_candidate["tokens"], original_candidate["beams"]
        baseline_test, baseline_predictions = evaluate(runner, test)

    save_report(output / "report.json", {
        "model": runner.spec.hub, "revision": runner.spec.revision, "dataset": "indosum", "source_tokens": 512,
        "counts": {"validation": valid_count, "test": test_count},
        "data_hash": common_digest({role: splits[role] for role in ("validation", "test")}),
        "training": training, "candidates": candidates, "selected": winner,
        "artifact": artifact.name if artifact else None,
        "baseline_test": baseline_test, "selected_test": selected_test,
        "baseline_test_predictions": baseline_predictions, "selected_test_predictions": selected_predictions,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--counts", type=int, nargs=3, default=(2000, 50, 100), metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()
    if min(*args.counts, args.epochs, args.accumulation) < 1 or args.patience < 0:
        parser.error("Jumlah contoh, epoch, dan akumulasi minimal 1. Patience minimal 0.")
    torch.set_num_threads(2)
    optimize(args.output, args.device, args.counts, args.epochs, args.accumulation, args.patience,
             prior_report=args.prior_report, train_lora=not args.skip_lora)


if __name__ == "__main__":
    main()
