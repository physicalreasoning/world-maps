"""Physion (Bear et al. 2021): object-contact prediction with per-stimulus human predictions.

Assets (public, fetched at run time):
  PhysionTest-Core zip (270 MB, S3): per scenario ``mp4s-redyellow`` (agent red, patient yellow; 512 px,
  30 fps, 5 s or longer; the outcome is shown), ``mp4s`` (random colours, shown to people), ``maps``
  (first-frame masks).
  Repository cogtoolslab/physics-benchmarking-neurips2021: ground-truth labels and the per-stimulus human
  accuracy and participant counts (1,144 of the 1,200 test stimuli).

Participants saw the first 1.5 s (45 frames) and answered "will the red object touch the yellow area?".
Here ``context`` = the first 45 frames (what people saw); ``outcome`` = the rest (ground truth for the
contact detector). Human P(yes) = accuracy when the label is True, 1 - accuracy when it is False.
"""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .paths import DATA

PHYSION = DATA / "physion"
CORE_URL = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip"
REPO_URL = "https://github.com/cogtoolslab/physics-benchmarking-neurips2021"
SCENARIOS = {
    "dominoes": "Dominoes",
    "support": "Support",
    "collide": "Collide",
    "contain": "Contain",
    "drop": "Drop",
    "roll": "Roll",
    "link": "Link",
    "drape": "Drape",
}
HUMAN_FILES = {
    "dominoes": "dominoes_pilot-production_1",
    "support": "towers_pilot-production_2",
    "collide": "collision_pilot-production_2",
    "contain": "containment_pilot-production_2",
    "drop": "drop_pilot-production_2",
    "roll": "rollingsliding_pilot-production_2",
    "link": "linking_pilot-production_2",
    "drape": "clothiness_pilot-production_2",
}
FPS = 30
CONTEXT_FRAMES = 45  # 1.5 s shown to participants


def fetch_assets() -> None:
    PHYSION.mkdir(parents=True, exist_ok=True)
    if not (PHYSION / "Physion").exists():
        z = PHYSION / "Physion.zip"
        if not z.exists():
            subprocess.run(["curl", "-L", "--retry", "3", "-o", str(z), CORE_URL], check=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(PHYSION, members=[m for m in zf.namelist() if not m.startswith("__MACOSX")])
    if not (PHYSION / "repo").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(PHYSION / "repo")], check=True)


@dataclass
class Stim:
    stim_id: str
    scenario: str
    label: bool
    human_acc: float | None
    n_human: int | None

    @property
    def p_yes_human(self) -> float | None:
        if self.human_acc is None:
            return None
        return self.human_acc if self.label else 1.0 - self.human_acc

    @property
    def video(self) -> Path:
        head, idx = self.stim_id.rsplit("_", 1)
        return (
            PHYSION
            / "Physion"
            / SCENARIOS[self.scenario]
            / "mp4s-redyellow"
            / f"{head}-redyellow_{idx}_img.mp4"
        )

    @property
    def map_png(self) -> Path:
        return PHYSION / "Physion" / SCENARIOS[self.scenario] / "maps" / f"{self.stim_id}_map.png"


def stimuli(with_human_only: bool = False) -> list[Stim]:
    repo = PHYSION / "repo"
    labels = pd.read_csv(repo / "stimuli" / "stimulus_names_and_labels.csv")
    human = {}
    for fname in HUMAN_FILES.values():
        f = repo / "results" / "csv" / "humans" / f"human_accuracy-{fname}_testing.csv"
        if f.exists():
            for _, r in pd.read_csv(f).iterrows():
                human[r.stim_ID] = (float(r.correct), int(r.c))
    out = []
    for _, r in labels.iterrows():
        h = human.get(r.stim_ID)
        s = Stim(
            stim_id=r.stim_ID,
            scenario=r.scenario,
            label=bool(r.label) if not isinstance(r.label, str) else r.label == "True",
            human_acc=h[0] if h else None,
            n_human=h[1] if h else None,
        )
        if with_human_only and h is None:
            continue
        out.append(s)
    return out


def stim_table(stims: list[Stim]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stim_id": s.stim_id,
                "scenario": s.scenario,
                "label": s.label,
                "human_acc": s.human_acc,
                "n_human": s.n_human,
                "p_yes_human": s.p_yes_human,
            }
            for s in stims
        ]
    )


def read_frames(path: Path, start: int = 0, stop: int | None = None, size: int | None = None) -> np.ndarray:
    """RGB uint8 frames [start, stop) of a video, optionally resized to size x size."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= start and (stop is None or i < stop):
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if size and fr.shape[0] != size:
                fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        i += 1
        if stop is not None and i >= stop:
            break
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, size or 512, size or 512, 3), np.uint8)


def sample_indices(n: int, k: int) -> np.ndarray:
    return np.linspace(0, n - 1, k).round().astype(int)


# ----------------------------------------------------------------------------- contact detector


class ColourModel:
    """Per-stimulus HSV model of the red agent and yellow patient.

    Calibrated on the first frame with the released segmentation map.
    """

    def __init__(self, frame0_rgb: np.ndarray, map_rgb: np.ndarray | None = None):
        hsv = cv2.cvtColor(frame0_rgb, cv2.COLOR_RGB2HSV).astype(int)
        if map_rgb is not None and map_rgb.shape[:2] != frame0_rgb.shape[:2]:
            map_rgb = cv2.resize(map_rgb, frame0_rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        if map_rgb is not None:
            red = (map_rgb[..., 0] > 200) & (map_rgb[..., 1] < 60) & (map_rgb[..., 2] < 60)
            yel = (map_rgb[..., 0] > 200) & (map_rgb[..., 1] > 200) & (map_rgb[..., 2] < 80)
        else:
            red = yel = np.zeros(frame0_rgb.shape[:2], bool)
        self.red = self._stats(hsv, red, default=(177, 230, 100))
        self.yellow = self._stats(hsv, yel, default=(26, 110, 210))
        self.map_yellow = yel if yel.sum() > 20 else None

    @staticmethod
    def _stats(hsv, mask, default):
        if mask.sum() < 20:
            return {
                "h": default[0],
                "s": default[1],
                "v": default[2],
                "s_lo": max(15, default[1] - 80),
                "v_lo": max(30, default[2] - 90),
            }
        h, s_, v = hsv[mask][:, 0], hsv[mask][:, 1], hsv[mask][:, 2]
        hm = int(np.median(h)) if default[0] < 90 else int(np.median(np.where(h < 90, h + 180, h))) % 180
        return {
            "h": hm,
            "s": int(np.median(s_)),
            "v": int(np.median(v)),
            "s_lo": int(max(15, np.percentile(s_, 3) - 15)),
            "v_lo": int(max(25, np.percentile(v, 3) - 25)),
        }

    def masks(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(int)
        h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        def near(h0, tol):
            d = np.abs(h - h0)
            return np.minimum(d, 180 - d) <= tol

        red = near(self.red["h"], 8) & (s_ >= max(self.red["s_lo"], 90)) & (v >= self.red["v_lo"])
        yellow = (
            near(self.yellow["h"], 5)
            & (s_ >= self.yellow["s_lo"])
            & (v >= self.yellow["v_lo"])
            & (np.abs(v - self.yellow["v"]) <= 90)
        )
        k = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        yellow = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        return red, yellow


def frame_measures(
    frame_rgb: np.ndarray, cm: ColourModel, dilate_px: int = 2, zone: np.ndarray | None = None
) -> dict:
    """Red-yellow geometry in one frame.

    Minimum distance (px at this resolution) and the fraction of the red object inside the dilated yellow
    region. ``zone`` (the yellow region seen during the context) is used in addition to the live yellow
    mask, because the red object hides the part of the zone it lands on.
    """
    red, yellow = cm.masks(frame_rgb)
    out = {
        "red_px": int(red.sum()),
        "yellow_px": int(yellow.sum()),
        "dist": float("nan"),
        "overlap": float("nan"),
    }
    ref = yellow if zone is None else (yellow | zone)
    if red.sum() < 10 or ref.sum() < 10:
        return out
    yd = cv2.dilate(ref.astype(np.uint8), np.ones((2 * dilate_px + 1,) * 2, np.uint8)) > 0
    inter = (red & yd).sum()
    # fraction of the red object inside the zone, or of the zone covered by red (a cloth covers a small zone)
    out["overlap"] = float(max(inter / red.sum(), inter / max(ref.sum(), 1)))
    if (red & ref).any():
        out["dist"] = 0.0
    else:
        dist = cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2, 3)
        out["dist"] = float(dist[red].min())
    return out


def context_zone(context: np.ndarray, cm: ColourModel) -> np.ndarray:
    """Union of the yellow masks over the context frames (the patient region before anything happens)."""
    z = np.zeros(context.shape[1:3], bool)
    for f in context[:: max(1, len(context) // 10)]:
        z |= cm.masks(f)[1]
    return z


def contact_score(
    frames: np.ndarray,
    cm: ColourModel,
    context: np.ndarray | None = None,
    touch_px: float = 2.0,
    overlap_gain: float = 0.10,
) -> dict:
    """Contact decision for an outcome sequence, relative to the end of the context (if given).

    contact = the red object comes within ``touch_px`` of the yellow region while it was farther than that
    at the end of the context, or its overlap with the yellow region grows by more than ``overlap_gain`` of
    its area. Distances are expressed at 512 px. Returns the decision, a soft score (overlap gain and
    distance drop), and coverage diagnostics.
    """
    scale = frames.shape[1] / 512.0
    zone = context_zone(context, cm) if context is not None and len(context) else None
    if zone is not None and zone.shape != frames.shape[1:3]:
        zone = cv2.resize(zone.astype(np.uint8), frames.shape[1:3][::-1], interpolation=cv2.INTER_NEAREST) > 0
    m = [frame_measures(f, cm, zone=zone) for f in frames]
    d = np.array([x["dist"] for x in m]) / scale
    ov = np.array([x["overlap"] for x in m])
    valid = ~np.isnan(d)
    base_d, base_ov = float("inf"), 0.0
    if context is not None and len(context):
        mc = [frame_measures(f, cm, zone=zone) for f in context[-10:]]
        dc = np.array([x["dist"] for x in mc]) / scale
        oc = np.array([x["overlap"] for x in mc])
        base_d = float(np.nanmin(dc)) if np.isfinite(dc).any() else float("inf")
        base_ov = float(np.nanmax(oc)) if np.isfinite(oc).any() else 0.0
    if not valid.any():
        return {
            "contact": None,
            "min_dist": float("nan"),
            "max_overlap": float("nan"),
            "frac_valid": 0.0,
            "score": float("nan"),
            "base_dist": base_d,
            "base_overlap": base_ov,
            "vanish": False,
            "first_contact_frame": None,
        }
    dm, om = float(np.nanmin(d)), float(np.nanmax(ov))
    touch = dm <= touch_px and (not np.isfinite(base_d) or base_d > touch_px)
    gain = om - base_ov > overlap_gain
    # the red object vanishes inside the zone (a container hides it): its area drops while it sits on the zone
    red_px = np.array([x["red_px"] for x in m], float)
    base_red = (
        float(np.mean([x["red_px"] for x in mc])) if context is not None and len(context) else float("nan")
    )
    vanish = bool(
        np.isfinite(base_red)
        and base_red > 50
        and red_px.min() < 0.4 * base_red
        and np.nanmax(ov[: int(np.argmin(red_px)) + 1]) > 0.5
    )
    contact = bool(touch or gain or vanish)
    hit = np.where(valid & ((d <= touch_px) | (ov - base_ov > overlap_gain)))[0]
    score = float(
        max(om - base_ov, 0.0) + max((min(base_d, 60.0) if np.isfinite(base_d) else 60.0) - dm, 0.0) / 60.0
    )
    return {
        "contact": contact,
        "min_dist": dm,
        "max_overlap": om,
        "frac_valid": float(valid.mean()),
        "score": score + (0.5 if vanish else 0.0),
        "base_dist": base_d,
        "base_overlap": base_ov,
        "vanish": vanish,
        "first_contact_frame": int(hit[0]) if len(hit) else None,
    }


def kinematic_features(frames: np.ndarray, cm: ColourModel) -> np.ndarray:
    """Heuristic features from the context frames.

    Last distance, distance change, approach speed toward the yellow region, red speed, mask areas and the
    relative offset. Used by the kinematic baseline readout.
    """
    ds, cr, cy = [], [], []
    for fr in frames:
        r, y = cm.masks(fr)
        ds.append(frame_measures(fr, cm)["dist"])
        cr.append(np.array(np.where(r)).mean(1)[::-1] if r.any() else np.array([np.nan, np.nan]))
        cy.append(np.array(np.where(y)).mean(1)[::-1] if y.any() else np.array([np.nan, np.nan]))
    ds, cr, cy = np.array(ds), np.array(cr), np.array(cy)
    last = np.nanmean(ds[-5:]) if np.isfinite(ds[-5:]).any() else 512.0
    first = np.nanmean(ds[:5]) if np.isfinite(ds[:5]).any() else 512.0
    v = np.nanmean(np.diff(cr, axis=0)[-10:], axis=0) if len(cr) > 10 else np.zeros(2)
    to_y = np.nan_to_num(cy[-1] - cr[-1])
    approach = float(np.dot(np.nan_to_num(v), to_y / (np.linalg.norm(to_y) + 1e-6)))
    r_last, y_last = cm.masks(frames[-1])
    return np.nan_to_num(
        np.array(
            [
                last,
                first - last,
                approach,
                float(np.linalg.norm(np.nan_to_num(v))),
                r_last.mean(),
                y_last.mean(),
                to_y[0],
                to_y[1],
            ],
            float,
        ),
        nan=0.0,
    )
