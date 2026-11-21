from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    transcript: str

    @property
    def text(self) -> str:
        return " ".join(self.transcript.split())


# Variasi ucapan dipakai untuk mendeteksi batas langkah. Teks sumber tetap utuh.
OPERATIONS = {
    "import": r"\b(?:import pandas|impor pandas)\b",
    "read_csv": r"\b(?:read[ _-]?csv|(?:membaca|baca|memuat|muat) (?:file |berkas |data )?csv)\b",
    "inspect": r"\b(?:head|info|describe)\s*\(|\b(?:lihat|periksa|tampilkan) (?:lima baris|data|isi data)\b",
    "isna": r"\b(?:is[ _-]?na|is[ _-]?null|cek nilai kosong|cek data kosong)\b",
    "dropna": r"\b(?:drop[ _-]?na|(?:hapus|menghapus) (?:baris|data|nilai) (?:yang )?(?:(?:dengan |mengandung )?\w+ )?kosong)\b",
    "fillna": r"\b(?:fill[ _-]?na|isi (?:nilai|data) kosong)\b",
    "drop_duplicates": r"\b(?:drop[ _-]?duplicates|(?:hapus|menghapus) (?:baris|data) duplikat)\b",
    "to_csv": r"\b(?:to[ _-]?csv|(?:simpan|ekspor)\b[^.!?]{0,50}\bcsv)\b",
}
ACTION = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in OPERATIONS.items()), re.IGNORECASE)
TRANSITION = re.compile(r"^(?:jadi\s*,?\s*)?(?:selanjutnya|kemudian|berikutnya|setelah itu|langkah (?:berikutnya|selanjutnya))\b", re.IGNORECASE)


def group_segments(segments: list[Segment]) -> list[Segment]:
    groups = []
    current: list[Segment] = []
    current_action = None
    word_count = 0
    for segment in segments:
        if not segment.text or not (math.isfinite(segment.start) and math.isfinite(segment.end)):
            continue
        if not 0 <= segment.start < segment.end:
            continue

        match = ACTION.search(segment.text)
        action = match.lastgroup if match else None
        words = len(segment.text.split())
        if current and (
            segment.start - current[-1].end >= 3
            or segment.end - current[0].start > 45
            or word_count + words > 180
            or (action and current_action and action != current_action)
            or (TRANSITION.match(segment.text) and action != current_action)
        ):
            groups.append(join_segments(current))
            current = []
            current_action = None
            word_count = 0
        current.append(segment)
        current_action = action or current_action
        word_count += words
    if current:
        groups.append(join_segments(current))
    return groups


def join_segments(segments: list[Segment]) -> Segment:
    return Segment(segments[0].start, segments[-1].end, "\n".join(segment.transcript for segment in segments))
