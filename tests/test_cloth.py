"""Unit tests for the shape-under-cloth helpers (no data download needed)."""

import numpy as np

from world_maps import cloth


def test_cloth_trial_blocks_alternate_same_and_different_category():
    trials = cloth.trials()
    assert len(trials) == 120
    assert sum(t.same_category for t in trials) == 60
    # blocks of six: even blocks same-category, odd blocks different-category
    assert all(t.same_category == ((t.t // 6) % 2 == 0) for t in trials)
    assert trials[3].same_category and not trials[7].same_category


def test_cloth_image_paths_follow_the_lab_layout():
    t = cloth.trials()[5]
    occ, un = t.images("occluded"), t.images("unoccluded")
    assert occ["target"].name == "trial_5_gtCloth.png" and occ["match"].name == "trial_5_gtNoCloth.png"
    assert (
        un["target"].name == "trial_5_gtTargetNoCloth.png"
        and un["distractor"].name == "trial_5_disNoCloth0.png"
    )


def test_pbas_rt_predictor_returns_first_iteration_at_criterion():
    anp = np.zeros((2, 32, 120, 200))
    anp[1, :, 7, 50:] = 1.0  # trial 7 (occluded) becomes correct from iteration 50
    rt = cloth.pbas_rt_predictor(anp, "occluded", 0.8)
    assert rt[7] == 51 and rt[0] == 200
