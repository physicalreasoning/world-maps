"""Exploratory: boundary AUROC by event type, and signs of the signal coefficients.

Descriptive only (no subgroup intervals). Writes results/events_exploratory.json.
"""

import json

import analyze_events as analyze
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from world_maps.paths import RESULTS

SIGS = ["vjepa_surprise", "dino_change", "flow_accel", "pix_diff"]

tr, _ = analyze.load_trials()
feat = pd.read_csv(RESULTS / "events_features.csv").dropna().reset_index(drop=True)
p = tr[tr.probe].merge(feat, on=["scene", "bin"])
p["t"] = p.bin * 0.2
p["t2"] = p.t**2
p["lab_pix"] = p.lab_pix.fillna(p.lab_pix.median())
p = analyze.zwithin(p, analyze.MODEL + analyze.LOW + ["lab_pix"])
p[["t", "t2"]] = (p[["t", "t2"]] - p[["t", "t2"]].mean()) / p[["t", "t2"]].std()
base = ["lab_pix", "pix_diff", "flow_speed", "flow_accel", "t", "t2"]

out = {}
for s in ["vjepa_surprise", "dino_change"]:
    m = LogisticRegression(C=1e4, max_iter=5000).fit(p[base + [s]].values, p.detected.values)
    out[s + "_coef"] = float(m.coef_[0][-1])

rows = []
for sc, g in feat.groupby("scene"):
    for etype, locs in analyze.B[sc].items():
        y = g.bin.isin([int(x) for x in locs]).astype(int).values
        if 0 < y.sum() < len(y):
            rows.append(dict(scene=sc, etype=etype, **{s: roc_auc_score(y, g[s].values) for s in SIGS}))
e = pd.DataFrame(rows)
out["auc_by_event_type"] = e.groupby("etype")[SIGS].mean().round(3).to_dict()
out["n_by_event_type"] = e.groupby("etype").size().to_dict()
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS / "events_exploratory.json", "w"), indent=1)
