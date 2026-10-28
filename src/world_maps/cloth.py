"""Shape under cloth (Yildirim, Siegel, Soltani, Ray Chaudhuri & Tenenbaum 2024, Nat Hum Behav).

Assets (public, fetched at run time):
  stimuli   github.com/CNCLgithub/intuitive-physics-3d-shape-perception-stimuli  (720 PNG, 224 x 224)
  behaviour github.com/CNCLgithub/PbAS-model-human-comparisons  (20,880 trials, 174 workers; PbAS runs)

Task: 2AFC match-to-sample. Trial t in 0..119 has a target (cloth-draped object in the occluded
condition, the bare object in another pose in the unoccluded condition), the matching object and a
distractor, both bare and in random poses. Trials come in blocks of six; even blocks are same-category
(harder), odd blocks different-category (easier), as in the lab's analysis code.

Human processing follows the lab's scripts: workers with overall accuracy <= 71/120 are excluded;
trial accuracy is the mean of ``correct`` per (condition, presentation time, trial); response times drop
|z| >= 3 within a worker-condition before averaging; participant bootstraps resample workers.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA

CLOTH = DATA / "cloth"
STIM = CLOTH / "stimuli"
BEH = CLOTH / "behaviour"
STIM_REPO = "https://github.com/CNCLgithub/intuitive-physics-3d-shape-perception-stimuli"
BEH_REPO = "https://github.com/CNCLgithub/PbAS-model-human-comparisons"
CATEGORIES = [
    "airplane",
    "bicycle",
    "bus",
    "car",
    "chair",
    "guitar",
    "motorcycle",
    "pistol",
    "rifle",
    "table",
]
TIMES = {"1000": "1s", "2000": "2s", "0": "unlimited"}
CONDITIONS = {0: "unoccluded", 1: "occluded"}
PBAS_ITER = {"1s": 48, "2s": 80, "unlimited": 110}  # best-matching iterations reported in the paper
PIXEL_PBAS_ITER = {"1s": 84, "2s": 116, "unlimited": 140}
ACC_THRESHOLD = 71.0 / 120


def fetch_assets() -> None:
    CLOTH.mkdir(parents=True, exist_ok=True)
    for repo, dest in [(STIM_REPO, STIM), (BEH_REPO, BEH)]:
        if not dest.exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", repo, str(dest)], check=True)


@dataclass
class Trial:
    t: int
    same_category: bool
    block: int

    def images(self, condition: str) -> dict[str, Path]:
        """Paths for target, match and distractor in one occlusion condition."""
        if condition == "occluded":
            d = STIM / "unoccluded-occluded"
            return {
                "target": d / f"trial_{self.t}_gtCloth.png",
                "match": d / f"trial_{self.t}_gtNoCloth.png",
                "distractor": d / f"trial_{self.t}_disNoCloth0.png",
            }
        d = STIM / "unoccluded-unoccluded"
        return {
            "target": d / f"trial_{self.t}_gtTargetNoCloth.png",
            "match": d / f"trial_{self.t}_gtNoCloth.png",
            "distractor": d / f"trial_{self.t}_disNoCloth0.png",
        }


def trials() -> list[Trial]:
    return [Trial(t=t, same_category=((t // 6) % 2 == 0), block=t // 6) for t in range(120)]


def image_index() -> pd.DataFrame:
    """All 720 stimulus images with (condition, trial, role, path)."""
    rows = []
    for tr in trials():
        for cond in CONDITIONS.values():
            for role, p in tr.images(cond).items():
                rows.append(
                    {
                        "condition": cond,
                        "trial": tr.t,
                        "role": role,
                        "path": str(p),
                        "same_category": tr.same_category,
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- behaviour


def load_behaviour() -> pd.DataFrame:
    df = pd.read_csv(BEH / "processed_data_anonymous.csv")
    df.columns = ["workerid", "condition", "stim_present", "stimulus", "response", "correct", "rt"]
    df["time"] = df["stim_present"].astype(str).map(TIMES)
    df["cond"] = df["condition"].map(CONDITIONS)
    acc = df.groupby("workerid")["correct"].mean()
    keep = acc[acc > ACC_THRESHOLD].index
    df = df[df.workerid.isin(keep)].copy()
    df["same_category"] = (df.stimulus // 6) % 2 == 0
    return df


def human_trial_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cond, time, trial): accuracy, n, mean RT after the |z| < 3 filter, mean log RT."""
    out = (
        df.groupby(["cond", "time", "stimulus"])
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    rts = []
    for (_w, _c, _t), g in df.groupby(["workerid", "cond", "time"]):
        z = (g.rt - g.rt.mean()) / (g.rt.std() + 1e-9)
        rts.append(g[np.abs(z) < 3])
    rt = (
        pd.concat(rts)
        .groupby(["cond", "time", "stimulus"])
        .agg(rt=("rt", "mean"), log_rt=("rt", lambda x: np.log(x).mean()))
        .reset_index()
    )
    out = out.merge(rt, on=["cond", "time", "stimulus"], how="left")
    out["same_category"] = (out.stimulus // 6) % 2 == 0
    return out.rename(columns={"stimulus": "trial"})


def participant_bootstrap(
    df: pd.DataFrame, cond: str, time: str, n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """Per-trial accuracy under participant resampling: array (n_boot, 120)."""
    sub = df[(df.cond == cond) & (df.time == time)]
    workers = sub.workerid.unique()
    mats = {
        w: g.set_index("stimulus")["correct"].reindex(range(120)).values for w, g in sub.groupby("workerid")
    }
    M = np.array([mats[w] for w in workers], float)  # workers x 120 (NaN where missing)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, 120))
    for b in range(n_boot):
        idx = rng.integers(0, len(workers), len(workers))
        out[b] = np.nanmean(M[idx], 0)
    return out


def split_half_ceiling(
    df: pd.DataFrame, cond: str, time: str, which=None, n_splits: int = 1000, seed: int = 0
) -> dict:
    """Bootstrapped split-half correlation of per-trial accuracy across participants (the lab's method)."""
    sub = df[(df.cond == cond) & (df.time == time)]
    workers = sub.workerid.unique()
    mats = {
        w: g.set_index("stimulus")["correct"].reindex(range(120)).values for w, g in sub.groupby("workerid")
    }
    M = np.array([mats[w] for w in workers], float)
    sel = np.arange(120) if which is None else np.array(which)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(len(workers))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        x, y = np.nanmean(M[a], 0)[sel], np.nanmean(M[b], 0)[sel]
        if x.std() > 0 and y.std() > 0:
            rs.append(np.corrcoef(x, y)[0, 1])
    rs = np.array(rs)
    return {
        "mean": float(rs.mean()),
        "ci": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
        "n_workers": int(len(workers)),
    }


# ----------------------------------------------------------------------------- PbAS reference


def pbas_arrays() -> dict[str, np.ndarray]:
    """Released per-run, per-trial, per-iteration correctness.

    Shape (2 conditions [unoccluded, occluded], 32 runs, 120 trials, 200 iterations).
    """
    return {
        "PbAS": np.load(BEH / "pbas-full-model-anp.npy"),
        "Pixel-PbAS": np.load(BEH / "td-no-image-encoding-anp.npy"),
    }


def pbas_trial_accuracy(anp: np.ndarray, cond: str, iteration: int) -> np.ndarray:
    c = 0 if cond == "unoccluded" else 1
    return anp[c, :, :, iteration].mean(0)


def pbas_rt_predictor(
    anp: np.ndarray, cond: str, criterion: float, same_mask: np.ndarray | None = None
) -> np.ndarray:
    """Earliest iteration at which the run-averaged accuracy exceeds the criterion (200 if never), per trial.

    The paper sets the criterion to the average participant accuracy of the trial's condition (same vs
    different category); pass ``criterion`` as a scalar or a length-120 array.
    """
    c = 0 if cond == "unoccluded" else 1
    acc = anp[c].mean(0)  # 120 x 200
    crit = np.broadcast_to(np.asarray(criterion, float), (120,))
    out = np.full(120, 200.0)
    for t in range(120):
        hit = np.where(acc[t] >= crit[t])[0]
        if len(hit):
            out[t] = hit[0] + 1
    return out
