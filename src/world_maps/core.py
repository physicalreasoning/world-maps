"""Small, dependency-light helpers shared by the scripts (unit-tested in CI)."""

from __future__ import annotations

import numpy as np

BIN_SECONDS = 0.2  # probe positions are 200 ms apart (lab notebook: iti = 200 ms)


def bin_onset_seconds(k: int) -> float:
    """Onset of slowdown probe ``k`` in original-video time (countdown excluded)."""
    return BIN_SECONDS * k


def letterbox_square(frame: np.ndarray) -> np.ndarray:
    """Pad a landscape frame to a square by replicating its top and bottom rows.

    Keeps the whole scene (no cropping). Portrait or square frames are returned unchanged.
    """
    h, w = frame.shape[:2]
    if w <= h:
        return frame
    top = (w - h) // 2
    bottom = w - h - top
    pad = [(top, bottom), (0, 0)] + [(0, 0)] * (frame.ndim - 2)
    return np.pad(frame, pad, mode="edge")


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def countdown_start(frames: np.ndarray, factor: float = 3.0) -> int:
    """Index of the first frame after a low-contrast grey countdown.

    The scene starts at the first frame whose contrast exceeds ``factor`` times the mean
    contrast of the first ten frames.
    """
    sd = frames.reshape(len(frames), -1).std(1)
    return int(np.argmax(sd > factor * sd[:10].mean()))
