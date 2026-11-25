"""Pinned models and generation for the three-model summarization experiment."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast

from src.summarization import INDONESIAN, guard_summary


@dataclass(frozen=True)
class ModelSpec:
    hub: str
    revision: str
    license: str
    vocabulary: str


MODELS = {
    "mbart": ModelSpec("facebook/mbart-large-50", "4ef55a20b36c6903b832e38f0604ab4bdf22c7d6", "MIT", "sentencepiece.bpe.model"),
    "mt5": ModelSpec("google/mt5-small", "73fb5dbe4756edadc8fbe8c769b0a109493acf7a", "Apache-2.0", "spiece.model"),
    "indobart": ModelSpec("indobenchmark/indobart", "73bead20e4a67f578f6f3b3f7038040304dc7065", "MIT", "sentencepiece.bpe.model"),
    "indobart-v2": ModelSpec("indobenchmark/indobart-v2", "7192ee75ba70ca247c7abfb8e7268588145c0bde", "MIT", "sentencepiece.bpe.model"),
}
MODEL_ROOT = Path(__file__).resolve().parents[1] / "data/raw/models"


def model_files(key, *, download=False):
    from huggingface_hub import snapshot_download

    spec = MODELS[key]
    names = ["*.json", spec.vocabulary, "pytorch_model.bin"]
    if key == "mbart":
        return Path(snapshot_download(spec.hub, revision=spec.revision, allow_patterns=names, local_files_only=not download))
    folder = MODEL_ROOT / spec.hub.split("/")[1]
    metadata = folder / ".cache/huggingface/download/pytorch_model.bin.metadata"
    if (folder / "pytorch_model.bin").is_file() and metadata.is_file():
        if metadata.read_text(encoding="utf-8").splitlines()[0] == spec.revision:
            return folder
    if not download:
        raise FileNotFoundError(f"Checkpoint belum tersedia pada revisi {spec.revision}: {spec.hub}")
    return Path(snapshot_download(spec.hub, revision=spec.revision, allow_patterns=names, local_dir=folder))


def load_tokenizer(key, path):
    if key == "mbart":
        return MBart50TokenizerFast.from_pretrained(path, src_lang=INDONESIAN, tgt_lang=INDONESIAN)
    if key in {"indobart", "indobart-v2"}:
        from indobenchmark import IndoNLGTokenizer

        return IndoNLGTokenizer.from_pretrained(path)
    return AutoTokenizer.from_pretrained(path, use_fast=False)


class Seq2SeqRunner:
    def __init__(self, key, tokenizer, model, *, source_tokens=256, target_tokens=72):
        self.key = key
        self.spec = MODELS[key]
        self.tokenizer = tokenizer
        self.model = model
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.checkpoint = "base"
        self.summary_language = INDONESIAN

    def set_summary_language(self, language):
        if language != INDONESIAN and self.key in {"indobart", "indobart-v2"}:
            raise ValueError(f"{self.key} tidak memiliki token bahasa {language}.")
        self.summary_language = language
        if self.key == "mbart":
            self.tokenizer.src_lang = language
            self.tokenizer.tgt_lang = language

    @property
    def identity(self):
        return f"{self.spec.hub}@{self.spec.revision} · {self.checkpoint}"

    def encode(self, text, reference=None):
        tokenizer = self.tokenizer
        if self.key in {"indobart", "indobart-v2"}:
            values = tokenizer.prepare_input_for_generation(text)["input_ids"]
            truncated = len(values) > self.source_tokens
            values = values[:self.source_tokens - 2] + values[-2:] if truncated else values
            batch = {"input_ids": torch.tensor([values]), "attention_mask": torch.ones((1, len(values)), dtype=torch.long)}
            if reference is not None:
                tokens = tokenizer(reference, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                tokens = tokens[:self.target_tokens - 2]
                start = [tokenizer.indonesian_token_id, tokenizer.bos_token_id]
                batch["decoder_input_ids"] = torch.tensor([start + tokens])
                batch["labels"] = torch.tensor([[tokenizer.bos_token_id, *tokens, tokenizer.eos_token_id]])
            return batch, truncated

        source = text
        if self.key == "mt5":
            prefix = "ringkas" if self.summary_language == INDONESIAN else "summarize"
            source = f"{prefix}: {text}"
        uncut = tokenizer(source, return_tensors="pt", verbose=False)
        truncated = uncut["input_ids"].shape[-1] > self.source_tokens
        batch = tokenizer(source, return_tensors="pt", max_length=self.source_tokens, truncation=True) if truncated else uncut
        if reference is not None:
            batch["labels"] = tokenizer(
                text_target=reference, return_tensors="pt", truncation=True, max_length=self.target_tokens,
            ).input_ids
        return batch, truncated

    @torch.inference_mode()
    def generate(self, batch):
        inputs = {key: batch[key].to(self.model.device) for key in ("input_ids", "attention_mask")}
        settings = {}
        if self.key == "mbart":
            settings = {
                "decoder_start_token_id": self.tokenizer.eos_token_id,
                "forced_bos_token_id": self.tokenizer.lang_code_to_id[self.summary_language],
            }
        elif self.key in {"indobart", "indobart-v2"}:
            settings = {
                "decoder_start_token_id": self.tokenizer.indonesian_token_id,
                "forced_bos_token_id": self.tokenizer.bos_token_id,
            }
        output = self.model.generate(
            **inputs, **settings, max_new_tokens=self.target_tokens, num_beams=2,
            do_sample=False, no_repeat_ngram_size=3, return_dict_in_generate=False,
        )
        return " ".join(self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True).split())

    def predict(self, text, *, required_clause=""):
        batch, truncated = self.encode(text)
        result = guard_summary(text, self.generate(batch), required_clause=required_clause)
        return result, truncated

    def summarize(self, text, *, required_clause=""):
        return self.predict(text, required_clause=required_clause)[0]


def load_runner(key, *, device="cpu", source_tokens=256, target_tokens=72, adapter=None, download=False, dtype=None):
    path = model_files(key, download=download)
    device = torch.device(device)
    if key == "mbart" and (device.type == "cuda" or dtype == torch.bfloat16):
        from scripts.train_mbart_pilot import load_base_model

        model = load_base_model(MODELS[key].revision, device=device, dtype=dtype or torch.bfloat16)
    else:
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path, dtype=dtype, use_safetensors=False, local_files_only=True, attn_implementation="eager",
        ).to(device)
    model.config._name_or_path = MODELS[key].hub
    tokenizer = load_tokenizer(key, path)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    runner = Seq2SeqRunner(key, tokenizer, model.eval(), source_tokens=source_tokens, target_tokens=target_tokens)
    if adapter is not None:
        runner.checkpoint = f"LoRA/{Path(adapter).parent.name}"
    return runner


def load_deployment(path: Path, *, device="cpu"):
    config = json.loads(path.read_text(encoding="utf-8"))
    key = config["model"]
    if config["revision"] != MODELS[key].revision:
        raise ValueError("Revisi checkpoint deployment berbeda dari revisi eksperimen.")
    adapter = path.parent / config["adapter"] if config["adapter"] else None
    partial = config.get("partial_weights")
    runner = load_runner(key, device=device, source_tokens=config["source_tokens"],
                         target_tokens=config["target_tokens"], adapter=adapter,
                         **({"dtype": torch.bfloat16} if partial else {}))
    if partial:
        from scripts.notebook_training import load_mbart_weights

        load_mbart_weights(runner, path.parent / partial)
    return runner
