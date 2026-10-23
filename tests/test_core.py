import numpy as np
import pytest

from world_maps import core


def test_bin_onset_matches_lab_notebook():
    # lab notebook: RT = (press - countdown) - k * 200 ms, so probe k starts at 0.2 k s
    assert core.bin_onset_seconds(0) == 0.0
    assert core.bin_onset_seconds(25) == pytest.approx(5.0)


def test_letterbox_square_keeps_scene_and_replicates_edges():
    frame = np.arange(2 * 6 * 3, dtype=np.uint8).reshape(2, 6, 3)
    sq = core.letterbox_square(frame)
    assert sq.shape == (6, 6, 3)
    # original rows are intact in the middle
    assert (sq[2:4] == frame).all()
    # padding replicates the nearest edge row
    assert (sq[0] == frame[0]).all() and (sq[-1] == frame[-1]).all()


def test_letterbox_square_leaves_square_frames():
    frame = np.zeros((4, 4), dtype=np.uint8)
    assert core.letterbox_square(frame) is frame


def test_letterbox_matches_opencv_replicate():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
    pad = (16 - 9) // 2
    ref = cv2.copyMakeBorder(frame, pad, 16 - 9 - pad, 0, 0, cv2.BORDER_REPLICATE)
    assert (core.letterbox_square(frame) == ref).all()


def test_holm_adjust_monotone_and_capped():
    adj = core.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)  # monotone: cannot fall below the previous step
    assert adj["d"] == pytest.approx(0.5)
    assert max(adj.values()) <= 1.0


def test_countdown_start_finds_scene_onset():
    rng = np.random.default_rng(1)
    grey = np.full((10, 8, 8), 128.0) + rng.normal(0, 1, (10, 8, 8))
    scene = rng.normal(128, 40, (5, 8, 8))
    assert core.countdown_start(np.concatenate([grey, scene])) == 10
