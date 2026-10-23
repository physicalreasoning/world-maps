"""Verify that slowdown probe k starts at t = 0.2*k s of the base video (pilot time alignment)."""

import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.argv = sys.argv[:1]
import extract_event_signals as extract  # noqa: E402

from world_maps.core import countdown_start  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402


def gray(path, skip=False):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (192, 108)).astype(np.float32))
    fr = np.array(fr)
    if skip:
        fr = fr[countdown_start(fr) :]
    return fr


res = []
for scene in sorted(extract.LEN):
    bpath = extract.base_video(scene)
    base = gray(
        bpath, skip=os.path.basename(bpath).startswith("final_")
    )  # strip countdown as the pipeline does
    if os.path.basename(bpath).startswith("final_"):
        base = np.repeat(base, 4, axis=0)  # 30 fps fallback -> 120 fps grid, as in extract.load_frames
    for k in [10, 25]:
        f = glob.glob(f"{extract.ROOT}/{scene}/final_{scene}_125ms_{k}.mp4")
        if not f:
            continue
        pr = gray(f[0], skip=True)
        lag = np.full(len(pr), np.nan)
        for i in range(len(pr)):
            c = 4 * i
            lo, hi = max(0, c - 40), min(len(base), c + 8)
            if hi - lo < 10:
                if c > len(base):
                    break
                continue
            win = base[lo:hi]
            mask = (win.max(0) - win.min(0)) > 20  # pixels that move in this window
            if mask.sum() < 30:
                continue
            dd = np.abs(win[:, mask] - pr[i][mask]).mean(1)
            lag[i] = c - (lo + int(np.argmin(dd)))
        ok = np.where(np.isfinite(lag))[0]
        on = [i for i in ok if lag[i] >= 2 and np.nanmean(lag[i : i + 6] >= 2) > 0.8]
        res.append(dict(scene=scene, k=k, expected=0.2 * k, observed=(on[0] / 30 if on else None)))
d = np.array([r["observed"] - r["expected"] for r in res if r["observed"] is not None])

per_scene = Counter(r["scene"] for r in res)
summ = dict(
    n_checked=len(res),
    n_scenes=len(per_scene),
    scenes_with_two=sum(v == 2 for v in per_scene.values()),
    n_detected=int(len(d)),
    median_diff_s=float(np.median(d)),
    within_0p1s=int((np.abs(d) <= 0.1).sum()),
    within_0p2s=int((np.abs(d) <= 0.2).sum()),
    outliers=[r for r in res if r["observed"] is not None and abs(r["observed"] - r["expected"]) > 0.2],
)
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, rows=res), open(RESULTS / "events_alignment.json", "w"), indent=1)
