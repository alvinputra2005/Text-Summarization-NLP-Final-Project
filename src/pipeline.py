from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.media import MediaError, prepare_video
from src.segmentation import group_segments
from src.summarization import load_indobart, model_reference, summarize
from src.transcription import transcribe


@dataclass(frozen=True)
class Step:
    start: float
    end: float
    transcript: str
    summary: str
    is_excerpt: bool


@dataclass(frozen=True)
class Guide:
    video: bytes
    steps: list[Step]
    model: str


def build_guide(video: bytes, progress: Callable[[float, str], None]) -> Guide:
    with TemporaryDirectory(prefix="tutorial-summarizer-") as scratch:
        upload = Path(scratch) / "upload.mp4"
        upload.write_bytes(video)
        progress(0.05, "Memeriksa MP4 dan jalur audio…")
        playable = prepare_video(upload)

        progress(0.22, "Mentranskripsi narasi dengan Whisper…")
        segments = transcribe(playable)
        progress(0.58, "Mengelompokkan transkrip per langkah…")
        groups = group_segments(segments)
        if not groups:
            raise MediaError("Narasi belum terdeteksi. Coba video dengan suara yang lebih jelas.")

        progress(0.64, "Memuat IndoBART-v2…")
        reference = model_reference()
        tokenizer, model = load_indobart(reference)
        steps = []
        for index, group in enumerate(groups):
            progress(0.66 + 0.32 * index / len(groups), f"Meringkas langkah {index + 1} dari {len(groups)}…")
            result = summarize(group.text, tokenizer, model)
            steps.append(Step(group.start, group.end, group.transcript, result.text, result.is_excerpt))
        return Guide(video if playable == upload else playable.read_bytes(), steps, reference)
