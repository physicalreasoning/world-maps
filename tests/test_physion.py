"""Unit tests for the Physion helpers on synthetic frames (no data download)."""

import numpy as np

from world_maps import physion as P


def _frame(red_xy, yellow_box=(300, 400, 420, 460), size=512):
    img = np.full((size, size, 3), 90, np.uint8)
    x0, x1, y0, y1 = yellow_box
    img[y0:y1, x0:x1] = (235, 215, 90)  # pale yellow zone
    cx, cy = red_xy
    img[cy - 15 : cy + 15, cx - 15 : cx + 15] = (170, 30, 30)  # dark red block
    return img


def _map(red_xy, yellow_box=(300, 400, 420, 460), size=512):
    m = np.zeros((size, size, 3), np.uint8)
    x0, x1, y0, y1 = yellow_box
    m[y0:y1, x0:x1] = (255, 255, 0)
    cx, cy = red_xy
    m[cy - 15 : cy + 15, cx - 15 : cx + 15] = (255, 0, 0)
    return m


def test_colour_model_recovers_red_and_yellow_masks():
    f0 = _frame((100, 200))
    cm = P.ColourModel(f0, _map((100, 200)))
    red, yellow = cm.masks(f0)
    assert 700 <= red.sum() <= 1000 and 3500 <= yellow.sum() <= 4500


def test_contact_score_detects_a_block_that_moves_onto_the_zone():
    f0 = _frame((100, 200))
    cm = P.ColourModel(f0, _map((100, 200)))
    context = np.stack([_frame((100 + 2 * t, 200 + 2 * t)) for t in range(10)])
    hit = np.stack([_frame((120 + 20 * t, 220 + 20 * t)) for t in range(12)])  # ends inside the zone
    miss = np.stack([_frame((120 + 2 * t, 220)) for t in range(12)])  # stays far away
    assert P.contact_score(hit, cm, context=context)["contact"] is True
    assert P.contact_score(miss, cm, context=context)["contact"] is False


def test_sample_indices_cover_the_range():
    idx = P.sample_indices(45, 16)
    assert idx[0] == 0 and idx[-1] == 44 and len(idx) == 16 and np.all(np.diff(idx) >= 0)


def test_p_yes_human_follows_the_label():
    s = P.Stim("x_0001", "dominoes", True, 0.8, 90)
    t = P.Stim("x_0002", "dominoes", False, 0.8, 90)
    assert s.p_yes_human == 0.8 and abs(t.p_yes_human - 0.2) < 1e-9
