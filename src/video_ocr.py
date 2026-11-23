"""Read timestamped editor frames and verify supported Python syntax locally."""

import ast
import base64
from collections import defaultdict
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps

from src.gemini_preprocessing import LEADING_FILLERS, SPOKEN_FORMS, TOKENS
from src.youtube import youtube_watch_url


CONDITION = re.compile(r"if\s+([a-zA-Z_]\w*)\s*(==|!=|<=|>=|<|>)\s*([+-]?\d{1,9}|[a-zA-Z_]\w*)\s*:")
LOOP = re.compile(r"(?:while|for)\s+.+:")
BREAK = re.compile(r"break(?:\s*#.*)?")
COMMENTED_BREAK = re.compile(r"#\s*break\b.*")
BRANCH_HEADER = re.compile(r"(?:(if|elif)\s+(.+?)|(else))\s*:")


@dataclass(frozen=True)
class OCRLine:
    text: str
    x: int
    y: int
    confidence: float


@dataclass(frozen=True)
class ScreenEvidence:
    expression: str
    condition: str
    loop: str
    action: str
    active: bool
    confidence: float


@dataclass(frozen=True)
class BranchEvidence:
    header: str
    conditions: tuple[str, ...]
    action: str


def find_branches(lines: list[OCRLine]) -> list[BranchEvidence]:
    branches = []
    for index, line in enumerate(lines):
        header = line.text.strip()
        match = BRANCH_HEADER.fullmatch(header)
        if not match or line.confidence < 65:
            continue
        kind = match[1] or match[3]
        prior = []
        if kind != "if":
            for previous in reversed(lines[:index]):
                if previous.x > line.x + 8 or previous.text.lstrip().startswith("#"):
                    continue
                same_level = abs(previous.x - line.x) <= 8
                sibling = BRANCH_HEADER.fullmatch(previous.text.strip()) if same_level else None
                if not sibling or previous.confidence < 65 or sibling[3]:
                    break
                prior.append(previous.text.strip())
                if sibling[1] == "if":
                    break
            if not prior or not prior[-1].startswith("if "):
                continue
        following = next((row for row in lines[index + 1:] if not row.text.lstrip().startswith("#")), None)
        if not following or following.x <= line.x + 8 or following.confidence < 65:
            continue
        action = following.text.strip()
        if len(action) > 100 or "`" in header or "`" in action:
            continue
        conditions = tuple(reversed(prior))
        program = "".join(f"{item}\n    pass\n" for item in conditions) + f"{header}\n    {action}"
        try:
            ast.parse(program)
            body = ast.parse(action).body
        except (SyntaxError, ValueError):
            continue
        if len(body) != 1:
            continue
        statement = body[0]
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) or isinstance(
            statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise)
        ):
            branches.append(BranchEvidence(header, conditions, action))
    return branches


def clean_cues(cues, start, end):
    trace = []
    previous = None
    for cue in sorted(cues, key=lambda item: item.start):
        if not (cue.start < end and cue.end > start) or not cue.text:
            continue
        content = LEADING_FILLERS.sub("", cue.text)

        def normalize_spoken(match):
            preceding = content[:match.start()]
            following = content[match.end():]
            if re.search(r"\b(?:variabel|kolom|fungsi|parameter)\s+$", preceding, re.I) or re.match(
                r"\s*(?:[=<>]|!=|[(:]|\+=|-=)", following,
            ):
                return match[0]
            return SPOKEN_FORMS.get(match[0].casefold(), match[0])

        text = TOKENS.sub(normalize_spoken, content)
        if re.fullmatch(r"(?:oke|nah|eh|ya|hmm)[.!?]*", text, re.I):
            text, status = "", "pengisi mandiri"
        elif previous and previous["cleaned"] == text and previous["end"] >= cue.start:
            text, status = "", "duplikat tumpang tindih"
        else:
            status = "dirapikan" if text != cue.transcript else "dipertahankan"
        row = {"start": cue.start, "end": cue.end, "original": cue.transcript, "cleaned": text, "status": status}
        trace.append(row)
        if text:
            previous = row
    return trace


def capture_editor(video, second, roi):
    data = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-ss", str(second), "-i", str(video),
        "-frames:v", "1", "-an", "-f", "image2pipe", "-c:v", "png", "pipe:1",
    ], check=True, capture_output=True, timeout=40).stdout
    with Image.open(BytesIO(data)) as image:
        x1, y1, x2, y2 = roi
        bounds = (round(x1 * image.width), round(y1 * image.height),
                  round(x2 * image.width), round(y2 * image.height))
        return image.crop(bounds).convert("RGB")


def tesseract_path():
    binary = os.environ.get("TESSERACT_EXE", "").strip() or shutil.which("tesseract")
    if not binary:
        windows = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR/tesseract.exe"
        if windows.is_file():
            binary = str(windows)
    if not binary:
        raise FileNotFoundError("Tesseract belum tersedia. Pasang Tesseract atau atur TESSERACT_EXE.")
    return binary


def preprocess_editor(editor):
    gray = ImageOps.grayscale(editor)
    if sum(gray.histogram()[:128]) > gray.width * gray.height / 2:
        gray = ImageOps.invert(gray)
    factor = max(1, min(4, 1_050 // gray.width))
    enlarged = gray.resize((gray.width * factor, gray.height * factor), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(enlarged, cutoff=1)


def image_payload(image, *, format="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def tesseract_lines(image, binary):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    result = subprocess.run([
        binary, "stdin", "stdout", "--psm", "6", "-l", "eng", "tsv",
    ], input=buffer.getvalue(), capture_output=True, check=True, timeout=20,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    words = defaultdict(list)
    for item in csv.DictReader(StringIO(result.stdout.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if item["level"] == "5" and item["text"].strip():
            key = tuple(item[name] for name in ("page_num", "block_num", "par_num", "line_num"))
            words[key].append(item)
    lines = [OCRLine(" ".join(word["text"] for word in row), min(int(word["left"]) for word in row),
                     min(int(word["top"]) for word in row), min(float(word["conf"]) for word in row))
             for row in words.values()]
    return sorted(lines, key=lambda line: (line.y, line.x))


def find_evidence(lines):
    matches = []
    for index, line in enumerate(lines):
        condition = CONDITION.fullmatch(line.text.strip())
        if not condition or line.confidence < 65:
            continue
        parent = next((previous for previous in reversed(lines[:index]) if previous.x + 8 < line.x), None)
        if not parent or not LOOP.fullmatch(parent.text.strip()) or parent.confidence < 65:
            continue
        expression = " ".join(condition.groups())
        body_x = None
        for child in lines[index+1:index+5]:
            if child.x <= line.x + 8:
                break
            if body_x is None:
                body_x = child.x
            if abs(child.x - body_x) > 8:
                continue
            action = child.text.strip()
            active = bool(BREAK.fullmatch(action))
            if child.confidence < 65 or (not active and not COMMENTED_BREAK.fullmatch(action)):
                continue
            try:
                parsed = ast.parse(f"{parent.text.strip()}\n    if {expression}:\n        break")
            except (SyntaxError, ValueError):
                continue
            if not isinstance(parsed.body[0], (ast.While, ast.For)):
                continue
            matches.append(ScreenEvidence(expression, line.text.strip(), parent.text.strip(),
                                          action, active, min(parent.confidence, line.confidence, child.confidence)))
            break
    return matches


def download_video(url, directory):
    url = youtube_watch_url(url)
    video_id = parse_qs(urlsplit(url).query)["v"][0]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{video_id}.mp4"
    if destination.is_file() and destination.stat().st_size:
        return destination

    with TemporaryDirectory(dir=directory) as folder:
        download = Path(folder) / "source.mp4"
        command = [
            sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
            "--js-runtimes", "node", "--match-filter", "!is_live", "--socket-timeout", "15",
            "--retries", "2", "--no-progress", "--max-filesize", "150M",
            "--format", "b[ext=mp4][height<=480]", "--output", str(download.resolve()), "--", url,
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=240)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Unduhan YouTube melewati batas waktu. Gunakan MP4 lokal melalui --video.") from error
        if result.returncode or not download.is_file() or not download.stat().st_size:
            raise RuntimeError("Video publik belum bisa diambil. Gunakan MP4 lokal melalui --video.")
        download.replace(destination)
    return destination
