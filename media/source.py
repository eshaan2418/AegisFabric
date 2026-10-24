"""Synthetic A/V source: chart screen-share video + spoken client questions (macOS `say`, else tones).

Everything is derived from `data/synth/finance.py` with a seed; speech audio is cached in
`data/generated/media/` (gitignored).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.synth import finance as F

CACHE = Path(__file__).resolve().parent.parent / "data" / "generated" / "media"
SR = 48000
FPS = 30


@dataclass
class Question:
    text: str
    intent: str
    start_s: float
    end_s: float


@dataclass
class Source:
    frames: np.ndarray  # (T, H, W, 3) uint8 at FPS
    audio: np.ndarray  # int16 mono at SR
    questions: list[Question]
    speech: str  # "say" | "tone"

    @property
    def duration_s(self) -> float:
        return len(self.frames) / FPS


def _say(text: str) -> np.ndarray | None:
    if not shutil.which("say"):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"q_{hashlib.sha1(text.encode()).hexdigest()[:12]}.wav"
    if not p.exists():
        subprocess.run(
            ["say", "-o", str(p), "--file-format=WAVE", f"--data-format=LEI16@{SR}", text],
            check=True,
            capture_output=True,
        )
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def _tone(seconds: float, seed: int) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = 220 + 40 * (seed % 5)
    return (8000 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.int16)


INTENTS = ["spending", "balance", "largest"]


def render_video(rng: np.random.Generator, st: F.Statement, n: int, size: int) -> np.ndarray:
    """Screen-share video of size x size: the statement text (scrolling slowly, top half), the running-balance
    chart (`finance.render_chart`, line dilated to 3 px so it survives 4:2:0 chroma) revealed left to right in
    the bottom half, a moving 'talking head' disc, and light temporal noise."""
    from PIL import Image, ImageDraw

    half = size // 2
    chart = F.render_chart(st, half)
    chart = np.kron(chart == 255, np.ones((1, 2), dtype=bool))[:, :size]  # stretch to full width
    line = chart.copy()
    for dy in (-1, 1):
        line |= np.roll(chart, dy, axis=0)
    # tall text panel rendered once, scrolled per frame
    lines = st.to_text().splitlines() * 3
    panel = Image.new("L", (size, 14 * len(lines) + half), 30)
    d = ImageDraw.Draw(panel)
    for i, ln in enumerate(lines):
        d.text((4, 2 + 14 * i), ln[:60], fill=210)
    panel = np.asarray(panel)
    yy, xx = np.mgrid[0:size, 0:size]
    frames = np.empty((n, size, size, 3), dtype=np.uint8)
    for f in range(n):
        fr = np.full((size, size, 3), 24, dtype=np.uint8)
        off = min(f // 3, panel.shape[0] - half)
        txt = panel[off : off + half]
        fr[:half] = txt[..., None]
        reveal = int((f + 1) / n * size)
        m = np.zeros((size, size), dtype=bool)
        m[half:] = line
        m[:, reveal:] = False
        fr[m] = (24, 230, 24)
        fr[-1, :] = 96
        r = size * 0.08 * (1 + 0.15 * np.sin(f / 2.0))
        cy, cx = size * 0.2 + 3 * np.sin(f / 5.0), size * 0.85 + 2 * np.cos(f / 7.0)
        fr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = (200, 160, 120)
        fr += rng.integers(0, 2, size=fr.shape, dtype=np.uint8)
        frames[f] = fr
    return frames


def build_source(
    seed: int, duration_s: float = 24.0, size: int = 256, q_starts=(2.5, 9.5, 16.5), use_say: bool = True
) -> Source:
    rng = np.random.default_rng(seed)
    st = F.make_statement(rng, n_tx=10)
    turns = [t for t in F.make_transcript(rng, st) if t["speaker"] == "client"]
    n = int(duration_s * FPS)
    frames = render_video(rng, st, n, size)
    audio = np.zeros(int(duration_s * SR) + SR, dtype=np.int16)
    questions = []
    speech = "say"
    for i, (turn, t0) in enumerate(zip(turns, q_starts, strict=False)):
        if t0 + 1.0 > duration_s:
            break
        pcm = _say(turn["text"]) if use_say else None
        if pcm is None:
            speech = "tone"
            pcm = _tone(2.5, i)
        pcm = pcm[: int(6.5 * SR)]
        a = int(t0 * SR)
        pcm = pcm[: len(audio) - a]
        audio[a : a + len(pcm)] = pcm
        # utterance end = last sample above a small energy floor
        nz = np.flatnonzero(np.abs(pcm) > 300)
        end = t0 + (nz[-1] if nz.size else len(pcm)) / SR
        questions.append(Question(turn["text"], INTENTS[i], t0, float(end)))
    return Source(frames, audio[: int(duration_s * SR)], questions, speech)
