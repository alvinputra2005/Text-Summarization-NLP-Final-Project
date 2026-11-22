"""Build a balanced, small code-summary sample and mix its train split with tutorial data."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path
import re

import pyarrow.parquet as pq


DATASET = "Fsoft-AIC/the-vault-function"
REVISION = "505c679056e49a2a269b64777ee7c496d22e1440"
LANGUAGES = ("c", "c_sharp", "cpp", "go", "java", "javascript", "php", "python", "ruby", "rust")
TOPICS = {
    "variables": re.compile(r"(?<![\w=<>!])(?:\$?[a-zA-Z_]\w*\s*:=|\$?[a-zA-Z_]\w*\s*=(?!=))"),
    "conditions": re.compile(r"\b(?:if|else|switch|match|unless|elsif)\b"),
    "loops": re.compile(r"\b(?:for|foreach|while|loop|each)\b"),
    "arrays_lists": re.compile(r"\[[^\]\n]*\]|\b(?:ArrayList|Array|List|Vec|append|slice|array)\b"),
    "functions": re.compile(r"\b(?:return|yield)\b"),
}
QUOTAS = {"train": tuple(TOPICS), "validation": ("conditions", "loops"), "test": ("variables", "arrays_lists")}
FOLDERS = {"train": "data/train/small", "validation": "data/validation", "test": "data/test"}
LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0", "Zlib"}
SOURCE_FIELDS = ("repo", "path", "hexsha", "license", "code", "short_docstring")
FIELDS = ("id", "text", "summary", "source", "source_lang", "target_lang", "topic", "language", "repo", "path", "hexsha", "licenses")


def load_examples(path, split, language, seen_code, repo_splits):
    required = set(QUOTAS[split])
    examples = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=list(SOURCE_FIELDS)):
        for row in batch.to_pylist():
            repo = (row["repo"] or "").strip()
            if not repo or repo_splits.get(repo, split) != split:
                continue
            code = (row["code"] or "").strip()
            summary = " ".join((row["short_docstring"] or "").split())
            licenses = row["license"] or []
            if not licenses or not set(licenses) <= LICENSES or not (45 <= len(code) <= 520):
                continue
            if not (4 <= len(summary.split()) <= 26) or not re.match(r"[A-Z]", summary):
                continue
            compact_code = " ".join(code.split())
            if summary.casefold().rstrip(".") in compact_code.casefold() or len(summary) >= len(compact_code):
                continue
            fingerprint = sha256(compact_code.encode("utf-8")).hexdigest()
            if fingerprint in seen_code:
                continue
            topic = next((name for name, pattern in TOPICS.items() if name in required and pattern.search(code)), None)
            if topic is None:
                continue
            seen_code.add(fingerprint)
            repo_splits[repo] = split
            required.remove(topic)
            examples.append({
                "id": f"vault:{language}:{fingerprint[:16]}", "text": code, "summary": summary.rstrip(".") + ".",
                "source": "the_vault", "source_lang": "en_XX", "target_lang": "en_XX", "topic": topic,
                "language": language, "repo": repo, "path": row["path"], "hexsha": row["hexsha"],
                "licenses": ";".join(licenses),
            })
            if not required:
                return examples
    raise ValueError(f"Contoh {language}/{split} kurang untuk topik: {', '.join(sorted(required))}")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Unduh satu shard per bahasa dan split dari The Vault.")
    args = parser.parse_args()
    raw = Path("data/raw/the-vault")
    if args.download:
        from huggingface_hub import snapshot_download

        patterns = [f"{folder}/{lang}-00000-of-*.parquet" for folder in FOLDERS.values() for lang in LANGUAGES]
        snapshot_download(DATASET, repo_type="dataset", revision=REVISION, local_dir=raw, allow_patterns=patterns, max_workers=5)
    data_root = Path("data/processed/multilingual_code")
    seen_code, repo_splits = set(), {}
    code_splits = {}
    for split, folder in FOLDERS.items():
        rows = []
        for language in LANGUAGES:
            candidates = sorted((raw / folder).glob(f"{language}-00000-of-*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"Shard {split}/{language} belum tersedia. Gunakan --download.")
            rows.extend(load_examples(candidates[0], split, language, seen_code, repo_splits))
        code_splits[split] = rows
        write_csv(data_root / f"{split}.csv", FIELDS, rows)
        print(f"The Vault {split}: {len(rows)} contoh dari {len(LANGUAGES)} bahasa", flush=True)

    for split in FOLDERS:
        with (Path("data/processed/coding_video_mix") / f"{split}.csv").open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fields = list(dict.fromkeys([*reader.fieldnames, *FIELDS]))
        if split == "train":
            rows.extend(code_splits["train"])
        write_csv(Path("data/processed/coding_multilingual_mix") / f"{split}.csv", fields, rows)
        print(f"Campuran tutorial {split}: {len(rows)} contoh", flush=True)


if __name__ == "__main__":
    main()
