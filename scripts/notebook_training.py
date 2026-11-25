"""Run the notebook's mixed Whisper and staged mBART experiments."""

import argparse
import json
from math import ceil
from pathlib import Path
import random
from statistics import mean


ROOT = Path("data/processed")
SEED = 42
SPLITS = ("train", "validation", "test")
AUDIO = {
    "LUMINA": ("lumina_split", SPLITS, "video_path", "speaker_id"),
    "NSS-ID": ("nss-id", ("train", "val", "test"), "audio_path", "speaker_id"),
    "Audio coding": ("whisper_video", SPLITS, "audio_path", "video_id"),
}


def save_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audio_plan(profile, *, corpora=None, full_splits=False):
    from scripts.notebook_asr import read_manifest

    names = tuple(AUDIO) if corpora is None else tuple(dict.fromkeys(corpora))
    caps = (2, 2, 2) if profile == "smoke" else (16, 8, 12)
    plan = {}
    holdout_ids = set()
    if "Audio coding" in names:
        holdout_ids = {row["id"] for row in json.loads(Path("samples/notebook_video_holdout.json").read_text("utf-8"))}
    for name in names:
        folder, aliases, media, group = AUDIO[name]
        selected = {}
        seen_groups = set()
        seen_media = set()
        for split, alias, cap in zip(SPLITS, aliases, caps, strict=True):
            frame = read_manifest(ROOT / folder / f"{alias}.csv", media)
            groups = set(frame[group])
            paths = set(frame["media"])
            if "" in groups or seen_groups & groups or seen_media & paths or (
                group == "video_id" and groups & holdout_ids
            ):
                raise ValueError(f"Pembagian media, penutur atau video bermasalah pada {name}/{split}.")
            seen_groups |= groups
            seen_media |= paths
            ordered = frame.sort_values("id")
            selected[split] = ordered if full_splits else ordered.sample(min(cap, len(frame)), random_state=SEED)
        plan[name] = selected
    return plan


def train_whisper(profile, output, device_name, *, epochs=None, accumulation=1, patience=2, corpora=None,
                  full_splits=False):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
    from scripts.notebook_asr import encode_clips, evaluate_asr

    device = torch.device(device_name)
    local = Path("data/raw/models/whisper-base-hf")
    model_id = str(local) if (local / "model.safetensors").is_file() else "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_id, language="id", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    plan = audio_plan(profile, corpora=corpora, full_splits=full_splits)
    encoded = {name: {split: encode_clips(frame, processor, model) for split, frame in splits.items()}
               for name, splits in plan.items()}
    training = [clip for splits in encoded.values() for clip in splits["train"]]
    model.requires_grad_(False)
    last = model.model.decoder.layers[-1]
    for attention in (last.self_attn, last.encoder_attn):
        attention.q_proj.requires_grad_(True)
        attention.v_proj.requires_grad_(True)
    params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-5, weight_decay=0.01, foreach=False)
    epochs = epochs if epochs is not None else (1 if profile == "smoke" else 5)
    updates = epochs * ceil(len(training) / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * updates), updates)
    print(f"Whisper · {device} · {len(training)} klip latih · {epochs} epoch", flush=True)
    original = snapshot(params)
    history, best_key, best_weights = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = random.Random(SEED + epoch).sample(training, len(training))
        losses = []
        for start in range(0, len(shuffled), accumulation):
            group = shuffled[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for _, features, labels in group:
                loss = model(**{key: value.to(device) for key, value in features.items()},
                             labels=labels.to(device), use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Loss Whisper tidak finite.")
                (loss / len(group)).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if len(losses) % 12 < len(group) or len(losses) == len(shuffled):
                print(f"Whisper epoch {epoch}/{epochs} · {len(losses)}/{len(shuffled)} klip", flush=True)
        valid = {name: evaluate_asr(model, processor, splits["validation"], device)[0]
                 for name, splits in encoded.items()}
        key = (mean(score["wer"] for score in valid.values()), mean(score["cer"] for score in valid.values()))
        history.append({"epoch": epoch, "loss": mean(losses), "macro_wer": key[0], "macro_cer": key[1],
                        "domains": valid, "lr_end": optimizer.param_groups[0]["lr"]})
        print(f"Whisper epoch {epoch} loss={mean(losses):.3f} macro-WER validasi={key[0]:.3f}", flush=True)
        if best_key is None or key < best_key:
            best_key, best_epoch = key, epoch
            best_weights = snapshot(params)
        if patience and epoch - best_epoch >= patience:
            break
    restore(params, best_weights)
    tuned = {name: evaluate_asr(model, processor, splits["test"], device) for name, splits in encoded.items()}
    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    restore(params, original)
    baseline = {name: evaluate_asr(model, processor, splits["test"], device) for name, splits in encoded.items()}
    predictions = {}
    for name in encoded:
        predictions[name] = {}
        for variant, evaluations in (("base", baseline), ("adapted", tuned)):
            frame = evaluations[name][1].astype(object)
            predictions[name][variant] = frame.where(frame.notna(), None).to_dict("records")
    return {"mode": profile, "base": "openai/whisper-base", "checkpoint": str(checkpoint), "device": device_name,
            "learning_rate": 1e-5, "max_epochs": epochs, "batch_size": 1, "accumulation": accumulation,
            "patience": patience, "best_epoch": best_epoch, "history": history,
            "splits": {name: {split: len(rows) for split, rows in parts.items()} for name, parts in plan.items()},
            "test": {name: {"base": baseline[name][0], "adapted": tuned[name][0]} for name in encoded},
            "test_predictions": predictions}


def snapshot(params):
    return {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in params.items()}


def restore(params, weights):
    import torch

    if set(params) != set(weights):
        raise ValueError("Parameter checkpoint tidak cocok dengan model.")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(weights[name])


def mbart_parameters(model):
    prefix = f"model.decoder.layers.{len(model.model.decoder.layers) - 1}.encoder_attn."
    names = {prefix + "q_proj.weight", prefix + "v_proj.weight"}
    return {name: parameter for name, parameter in model.named_parameters()
            if name in names}


def load_mbart_weights(runner, path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base"] != runner.spec.hub or payload["revision"] != runner.spec.revision:
        raise ValueError("Checkpoint memakai revisi mBART yang berbeda.")
    restore(mbart_parameters(runner.model), payload["params"])
    if "source_tokens" in payload:
        runner.source_tokens = payload["source_tokens"]
        runner.target_tokens = payload["target_tokens"]
    runner.checkpoint = f"{payload['strategy']} · {payload['mode']}"


def token_budget(tokenizer, splits):
    lengths = {}
    for dataset, partitions in splits.items():
        lengths[dataset] = {}
        for split, rows in partitions.items():
            lengths[dataset][split] = (
                [len(ids) for ids in tokenizer([row["text"] for row in rows], verbose=False).input_ids],
                [len(ids) for ids in tokenizer(text_target=[row["summary"] for row in rows], verbose=False).input_ids],
            )
    train_source, train_target = ([n for partitions in lengths.values() for n in partitions["train"][column]]
                                  for column in (0, 1))

    def cap(values, multiple, minimum, maximum):
        p90 = sorted(values)[ceil(0.90 * len(values)) - 1]
        return min(maximum, max(minimum, ceil(p90 / multiple) * multiple))

    source = cap(train_source, 64, 256, 512)
    target = cap(train_target, 16, 64, 96)
    audit = {name: {split: {"examples": len(inputs), "source_truncated": sum(n > source for n in inputs),
                            "target_truncated": sum(n > target for n in targets)}
                    for split, (inputs, targets) in partitions.items()}
             for name, partitions in lengths.items()}
    return source, target, audit


def fit_mbart(runner, params, training, validation, domain, epochs, *, accumulation=8, patience=2,
              selection=None, learning_rate=2e-5):
    import torch
    from scripts.compare_seq2seq import evaluate, score_key

    master = {name: torch.nn.Parameter(parameter.detach().float().clone()) for name, parameter in params.items()}
    optimizer = torch.optim.AdamW(master.values(), lr=learning_rate, weight_decay=0.01, foreach=False)
    history, best_key, best_weights = [], None, None
    for epoch in range(1, epochs + 1):
        runner.model.train()
        rows = random.Random(SEED + epoch).sample(training, len(training))
        losses = []
        for start in range(0, len(rows), accumulation):
            group = rows[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            runner.model.zero_grad(set_to_none=True)
            for _, batch, _ in group:
                batch = {key: tensor.to(runner.model.device) for key, tensor in batch.items()}
                loss = runner.model(**batch, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Loss {runner.key} tidak finite.")
                (loss / len(group)).backward()
                losses.append(loss.item())
            for name, parameter in params.items():
                if parameter.grad is not None:
                    master[name].grad = parameter.grad.detach().float()
            torch.nn.utils.clip_grad_norm_(master.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            restore(params, master)
            if len(losses) % 25 < len(group) or len(losses) == len(rows):
                print(f"{runner.key} {domain} epoch {epoch}/{epochs} · {len(losses)}/{len(rows)} contoh", flush=True)
        valid, _ = evaluate(runner, validation)
        key = selection(valid) if selection is not None else score_key(valid, domain)
        history.append({"epoch": epoch, "loss": mean(losses), "validation": valid})
        print(f"{runner.key} {domain} epoch {epoch} loss={mean(losses):.3f} ROUGE-L validasi={valid['rougeL']:.3f}", flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_weights = snapshot(params)
        if patience and epoch - best_epoch >= patience:
            break
    restore(params, best_weights)
    return {"dataset": domain, "history": history, "best_epoch": best_epoch}


def train_mbart(profile, output, device):
    import torch
    from scripts.compare_seq2seq import common_digest, encode_rows, evaluate, experiment_rows, score_key
    from src.seq2seq_models import load_runner

    raw = {name: experiment_rows(name, profile) for name in ("indosum", "coding_video_mix")}
    if profile == "smoke":
        raw = {name: {role: rows[:2] for role, rows in splits.items()} for name, splits in raw.items()}
    runner = load_runner("mbart", device=device, dtype=torch.bfloat16)
    source_tokens, target_tokens, length_audit = token_budget(runner.tokenizer, raw)
    runner.source_tokens, runner.target_tokens = source_tokens, target_tokens
    encoded = {name: {role: encode_rows(runner, rows) for role, rows in splits.items()}
               for name, splits in raw.items()}
    coding = encoded["coding_video_mix"]
    model = runner.model
    model.requires_grad_(False)
    params = mbart_parameters(model)
    if len(params) != 2:
        raise RuntimeError("Dua proyeksi cross-attention decoder terakhir diperlukan.")
    for parameter in params.values():
        parameter.requires_grad_(True)
    original = snapshot(params)
    epochs = 1 if profile == "smoke" else 5
    settings = {"base": runner.spec.hub, "revision": runner.spec.revision, "mode": profile,
                "source_tokens": source_tokens, "target_tokens": target_tokens}
    variants = {"base": {"validation": evaluate(runner, coding["validation"])[0], "stages": []}}
    weights = {"base": original}
    indosum = encoded["indosum"]
    first_stage = fit_mbart(runner, params, indosum["train"], indosum["validation"], "indosum", epochs)
    indosum_checkpoint = output / "indosum.pt"
    torch.save({**settings, "strategy": "indosum", "params": snapshot(params)}, indosum_checkpoint)
    for name in ("coding", "indosum_coding"):
        stages = []
        if name == "indosum_coding":
            load_mbart_weights(runner, indosum_checkpoint)
            stages.append(first_stage)
        else:
            restore(params, original)
        torch.manual_seed(SEED)
        stage = fit_mbart(runner, params, coding["train"], coding["validation"], "coding_video_mix", epochs)
        stages.append(stage)
        variants[name] = {"validation": stage["history"][stage["best_epoch"] - 1]["validation"], "stages": stages}
        weights[name] = snapshot(params)
    winner = max(variants, key=lambda name: score_key(variants[name]["validation"], "coding_video_mix"))
    for name, variant in variants.items():
        restore(params, weights[name])
        variant["test"] = evaluate(runner, coding["test"])[0]
    restore(params, original)
    indosum_base = evaluate(runner, indosum["test"])[0]
    load_mbart_weights(runner, indosum_checkpoint)
    indosum_tuned = evaluate(runner, indosum["test"])[0]
    checkpoint = output / "selected.pt"
    torch.save({**settings, "strategy": winner, "params": weights[winner]}, checkpoint)
    deployment = output / "deployment.json"
    save_report(deployment, {**settings, "model": "mbart", "adapter": None, "partial_weights": checkpoint.name})
    return {**settings,
            "device": device, "checkpoint": str(checkpoint), "deployment": str(deployment), "winner": winner, "variants": variants,
            "indosum_checkpoint": str(indosum_checkpoint),
            "indosum_test": {"base": indosum_base, "adapted": indosum_tuned},
            "length_audit": length_audit, "sample_hashes": {name: common_digest(rows) for name, rows in raw.items()},
            "learning_rate": 2e-5, "batch": 1, "accumulation": 8, "max_epochs": epochs,
            "parameter_count": sum(parameter.numel() for parameter in params.values()),
            "splits": {name: {split: len(rows) for split, rows in parts.items()} for name, parts in raw.items()}}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("whisper", "mbart"))
    parser.add_argument("--profile", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--patience", type=int, help="0 untuk menjalankan semua epoch.")
    parser.add_argument("--corpus", choices=tuple(AUDIO), action="append", help="Bisa diulang untuk memilih beberapa korpus.")
    parser.add_argument("--full-splits", action="store_true", help="Pakai semua contoh pada manifest Whisper.")
    args = parser.parse_args()
    if args.task == "mbart" and (args.full_splits or any(
        value is not None for value in (args.epochs, args.accumulation, args.patience, args.corpus)
    )):
        parser.error("Opsi ini untuk Whisper. Gunakan scripts.compare_partial_seq2seq untuk model ringkasan.")
    if ((args.epochs is not None and args.epochs < 1)
            or (args.accumulation is not None and args.accumulation < 1)
            or (args.patience is not None and args.patience < 0)):
        parser.error("Epoch dan akumulasi minimal 1. Patience minimal 0.")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.task == "whisper":
        report = train_whisper(args.profile, args.output, args.device, epochs=args.epochs,
                               accumulation=args.accumulation or 1,
                               patience=2 if args.patience is None else args.patience, corpora=args.corpus,
                               full_splits=args.full_splits)
    else:
        report = train_mbart(args.profile, args.output, args.device)
    save_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
