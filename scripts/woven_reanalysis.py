"""Design-based minimum detectable effect for model-vs-model human-choice comparisons in Woven,
using the public trial assignment (OSF 29jnd, CC0). Model choice probability per trial = share
of the 10 released model iterations choosing each option (add-0.5 smoothing)."""

import json
import sys

import numpy as np
import pandas as pd

from world_maps.paths import RESULTS, WOVEN

rng = np.random.default_rng(7)
METRIC = sys.argv[1] if len(sys.argv) > 1 else "agree"
out = {}
for task, f in [("stiffness", "parsed_questions_stiff.csv"), ("mass", "parsed_questions_mass.csv")]:
    d = pd.read_csv(
        WOVEN / f,
        usecols=[
            "trial",
            "subject",
            "iter_sub",
            "target",
            "match",
            "distract",
            "human_res",
            "human_acc",
            "woven_res",
            "dnn_res",
        ],
    )
    g = (
        d.groupby(["subject", "trial"])
        .agg(
            human=("human_res", "first"),
            acc=("human_acc", "first"),
            target=("target", "first"),
            match=("match", "first"),
            distract=("distract", "first"),
            pw=("woven_res", lambda x: ((x == 1).sum() + 0.5) / (len(x) + 1)),
            pd_=("dnn_res", lambda x: ((x == 1).sum() + 0.5) / (len(x) + 1)),
        )
        .reset_index()
    )

    def ls(p, h):
        return np.where(h == 1, np.log(p), np.log(1 - p))

    g["ls_w"] = ls(g.pw, g.human)
    g["ls_d"] = ls(g.pd_, g.human)
    g["diff"] = g.ls_w - g.ls_d
    g["agree_w"] = np.where(g.human == 1, g.pw, 1 - g.pw)
    g["agree_d"] = np.where(g.human == 1, g.pd_, 1 - g.pd_)
    subs = g.subject.unique()
    movies = pd.unique(g[["target", "match", "distract"]].values.ravel())
    res = {
        "n_participants": len(subs),
        "n_trials": len(g),
        "n_movies": len(movies),
        "human_acc": float(g.acc.mean()),
        "woven_minus_dnn_logscore_per_trial": float(g["diff"].mean()),
        "woven_minus_dnn_agreement": float((g.agree_w - g.agree_d).mean()),
    }
    S = {s: i for i, s in enumerate(subs)}
    M = {m: i for i, m in enumerate(movies)}
    si = g.subject.map(S).values
    ti, mi, di = [g[c].map(M).values for c in ["target", "match", "distract"]]
    x = (g.agree_w - g.agree_d).values if METRIC == "agree" else g["diff"].values
    for scheme in ["participants", "participants_x_target", "participants_x_all3"]:
        bs = []
        for _ in range(2000):
            ws = np.bincount(rng.integers(0, len(subs), len(subs)), minlength=len(subs))[si].astype(float)
            if scheme != "participants":
                wm = np.bincount(rng.integers(0, len(movies), len(movies)), minlength=len(movies)).astype(
                    float
                )
                ws = ws * (wm[ti] if scheme == "participants_x_target" else wm[ti] * wm[mi] * wm[di])
            bs.append((ws * x).sum() / ws.sum() if ws.sum() > 0 else np.nan)
        se = float(np.nanstd(bs))
        res[f"se_{scheme}"] = se
        res[f"mde80_{scheme}"] = 2.8 * se
        res[f"ci95_{scheme}"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    out[task] = res
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS / f"woven_{METRIC}.json", "w"), indent=1)
