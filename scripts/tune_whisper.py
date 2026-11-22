"""Select a Faster Whisper checkpoint and beam size on NSS-ID validation audio."""

import argparse
import gc
import json
import os
from pathlib import Path

from scripts.evaluate_media import evaluate_whisper
from src.transcription import load_faster_whisper, transcribe
from src.evaluation import speech_metrics


def configuration(model, beam):
    os.environ["WHISPER_BACKEND"] = "faster"
    os.environ["WHISPER_MODEL_ID"] = model
    os.environ["WHISPER_BEAM_SIZE"] = str(beam)
    load_faster_whisper.clear()
    gc.collect()


def run_evaluation(manifest, count, model, beam):
    configuration(model, beam)
    print(f"{model} · beam {beam} · {manifest.name}", flush=True)
    result = evaluate_whisper(manifest, count)
    print(f"WER {result['wer']:.2%} · CER {result['cer']:.2%}", flush=True)
    return result


def check_pandas_video():
    video = Path("outputs/demo-pandas-narasi-sintetis.mp4")
    if not video.is_file():
        return None
    reference = json.loads(Path("samples/pandas_synthetic_reference.json").read_text(encoding="utf-8"))["steps"]
    ref = " ".join(step["transcript"] for step in reference)
    predicted = " ".join(segment.text for segment in transcribe(video))
    return {**speech_metrics([ref], [predicted]), "prediction": predicted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", required=True)
    parser.add_argument("--turbo", help="Opsional, checkpoint Turbo untuk perbandingan ukuran model.")
    parser.add_argument("--previous-report", type=Path, help="Gunakan ulang pengukuran small dari laporan NSS-ID yang sama.")
    parser.add_argument("--output", type=Path, default=Path("outputs/whisper-decoding-pilot.json"))
    args = parser.parse_args()
    root = Path("data/processed/nss-id")
    models = (args.small, args.turbo) if args.turbo else (args.small,)
    settings = [(model, beam) for model in models for beam in (1, 3)]
    previous = json.loads(args.previous_report.read_text(encoding="utf-8")) if args.previous_report else {}
    recorded_validation = {
        (item["whisper"], item["beam_size"]): item for item in previous.get("validation", [])
        if Path(item["manifest"]) == root / "val.csv" and item["samples"] == 12
    }
    recorded_test = {
        (item["whisper"], item["beam_size"]): item for item in previous.get("test", {}).values()
        if Path(item["manifest"]) == root / "test.csv" and item["samples"] == 24
    }
    validation = [recorded_validation.get((model, beam)) or run_evaluation(root / "val.csv", 12, model, beam)
                  for model, beam in settings]
    selected = min(validation, key=lambda result: (result["wer"], result["cer"], result["transcription_seconds"]))
    winner = (selected["whisper"], selected["beam_size"])
    test = {}
    for label, (model, beam) in (("baseline", settings[0]), ("selected", winner)):
        if label == "selected" and settings[0] == winner:
            test[label] = test["baseline"]
            continue
        if (model, beam) in recorded_test:
            test[label] = recorded_test[(model, beam)]
        else:
            test[label] = run_evaluation(root / "test.csv", 24, model, beam)
            test[label]["synthetic_pandas"] = check_pandas_video()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump({"selected": {"model": winner[0], "beam_size": winner[1]}, "validation": validation, "test": test},
                  file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Laporan {args.output}", flush=True)


if __name__ == "__main__":
    main()
