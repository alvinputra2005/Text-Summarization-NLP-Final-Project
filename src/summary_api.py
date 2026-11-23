"""Local text and CSV inference using the notebook's selected checkpoint."""

import csv
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.evaluation import summary_metrics
from src.seq2seq_models import load_deployment


TEXT_LIMIT = 20_000


def predict_row(runner, text, reference=""):
    text = text.strip()
    if not text or len(text) > TEXT_LIMIT:
        raise ValueError(f"Teks harus berisi 1–{TEXT_LIMIT} karakter.")
    started = perf_counter()
    result, truncated = runner.predict(text)
    output = {
        "model": runner.spec.hub, "revision": runner.spec.revision, "checkpoint": runner.checkpoint,
        "summary": result.text, "is_excerpt": result.is_excerpt, "input_truncated": truncated,
        "latency_ms": round((perf_counter() - started) * 1_000, 2),
    }
    if reference.strip():
        output.update(summary_metrics(reference, result.text))
    return output


def summarize_csv(runner, source: Path, destination: Path):
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"id", "text"}.issubset(reader.fieldnames or ()):
            raise ValueError("CSV harus memiliki kolom id dan text.")
        rows = list(reader)
    ids = [(row.get("id") or "").strip() for row in rows]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("CSV harus berisi contoh dengan nilai id yang unik.")
    results = [
        {"id": key, **predict_row(runner, row.get("text") or "", row.get("reference_summary") or "")}
        for key, row in zip(ids, rows, strict=True)
    ]
    columns = list(dict.fromkeys(column for row in results for column in row))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    return results


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)
    reference_summary: str = Field(default="", max_length=TEXT_LIMIT)


@lru_cache(maxsize=1)
def configured_runner(path: str):
    return load_deployment(Path(path))


def create_app(runner=None):
    application = FastAPI(title="Ringkasan tutorial")
    lock = Lock()

    @application.post("/summarize")
    def summarize(request: SummarizeRequest):
        path = os.environ.get("SUMMARY_DEPLOYMENT", "").strip()
        if runner is None and not path:
            raise HTTPException(status_code=503, detail="Atur SUMMARY_DEPLOYMENT ke deployment.json hasil notebook.")
        with lock:
            try:
                model = runner if runner is not None else configured_runner(path)
            except FileNotFoundError:
                raise HTTPException(status_code=503, detail="Berkas deployment atau checkpoint belum tersedia.") from None
            try:
                return predict_row(model, request.text, request.reference_summary)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from None

    return application


app = create_app()
