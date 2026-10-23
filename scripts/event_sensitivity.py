"""Pilot 0 sensitivity analyses requested in review cycle 3 (run after the primary analysis).
S0  original: within-scene z-scoring (transductive on covariates, no labels), leave-one-scene-out
S1  training-fold-only scaling and imputation (no within-scene normalization), leave-one-scene-out
S2  S1 + leave-one-event-pair-family-out (e.g. both collision_collision videos held out together)
S3  S1 excluding the one 30-fps fallback video (falling_falling4316)
S4  S1 excluding bins without a full 0.8 s V-JEPA context (bin < 4)
Inference: scene (or family) bootstrap, and a crossed participant x scene bootstrap for S1.
Also: corrected descriptive counts and raw (unsmoothed) Woven agreement.
"""

import ast
import csv
import json

import analyze_events as analyze
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from world_maps.paths import EVENT_TRIALS, RESULTS, WOVEN

rng = np.random.default_rng(11)
tr, _ = analyze.load_trials()
feat = pd.read_csv(RESULTS / "events_features.csv").dropna().reset_index(drop=True)
p0 = tr[tr.probe].merge(feat, on=["scene", "bin"]).reset_index(drop=True)
p0["t"] = p0.bin * 0.2
p0["t2"] = p0.t**2
p0["family"] = p0.scene.str.replace(r"\d+$", "", regex=True)
BASE = ["lab_pix", "pix_diff", "flow_speed", "flow_accel", "t", "t2"]
SIGS = ["vjepa_surprise", "dino_change"]


def fold_ll(df, cols, group, scaling):
    ll = np.zeros(len(df))
    for g in df[group].unique():
        te = (df[group] == g).values
        trn, tst = df.loc[~te, cols].copy(), df.loc[te, cols].copy()
        if scaling == "train":
            med = trn.median()
            trn, tst = trn.fillna(med), tst.fillna(med)
            mu, sd = trn.mean(), trn.std() + 1e-12
            trn, tst = (trn - mu) / sd, (tst - mu) / sd
        m = LogisticRegression(C=1.0, max_iter=3000).fit(trn.values, df.loc[~te, "detected"].values)
        pr = np.clip(m.predict_proba(tst.values)[:, 1], 1e-6, 1 - 1e-6)
        y = df.loc[te, "detected"].values
        ll[te] = y * np.log(pr) + (1 - y) * np.log(1 - pr)
    return ll


def boot(df, diff, unit, n=2000):
    d = pd.Series(diff).groupby(df[unit].values).agg(["sum", "count"])
    u = d.index.values
    bs = [d.loc[s, "sum"].sum() / d.loc[s, "count"].sum() for s in (rng.choice(u, len(u)) for _ in range(n))]
    return [
        1000 * float(diff.mean()),
        1000 * float(np.percentile(bs, 2.5)),
        1000 * float(np.percentile(bs, 97.5)),
    ]


def crossed(df, diff, n=2000):
    subs, scs = df["sub"].unique(), df.scene.unique()
    si = df["sub"].map({s: i for i, s in enumerate(subs)}).values
    ci = df.scene.map({s: i for i, s in enumerate(scs)}).values
    bs = []
    for _ in range(n):
        w = (
            np.bincount(rng.integers(0, len(subs), len(subs)), minlength=len(subs))[si]
            * np.bincount(rng.integers(0, len(scs), len(scs)), minlength=len(scs))[ci]
        )
        bs.append((w * diff).sum() / w.sum())
    return [
        1000 * float(diff.mean()),
        1000 * float(np.percentile(bs, 2.5)),
        1000 * float(np.percentile(bs, 97.5)),
    ]


out = {}
# S0 (original, reproduced here)
z = analyze.zwithin(p0.copy(), analyze.MODEL + analyze.LOW + ["lab_pix"])
z["lab_pix"] = z.lab_pix.fillna(0)
z[["t", "t2"]] = (z[["t", "t2"]] - z[["t", "t2"]].mean()) / z[["t", "t2"]].std()
l0 = fold_ll(z, BASE, "scene", "none")
out["S0_within_scene_z"] = {s: boot(z, fold_ll(z, BASE + [s], "scene", "none") - l0, "scene") for s in SIGS}
# S1
l1 = fold_ll(p0, BASE, "scene", "train")
d1 = {s: fold_ll(p0, BASE + [s], "scene", "train") - l1 for s in SIGS}
out["S1_train_fold_scaling"] = {s: boot(p0, d1[s], "scene") for s in SIGS}
out["S1_crossed_participant_x_scene"] = {s: crossed(p0, d1[s]) for s in SIGS}
# S2
l2 = fold_ll(p0, BASE, "family", "train")
out["S2_family_holdout"] = {
    s: boot(p0, fold_ll(p0, BASE + [s], "family", "train") - l2, "family") for s in SIGS
}
out["S2_n_families"] = int(p0.family.nunique())
# S3, S4
for name, mask in [
    ("S3_drop_30fps_video", p0.scene != "falling_falling4316"),
    ("S4_full_context_only", p0.bin >= 4),
]:
    q = p0[mask].reset_index(drop=True)
    lq = fold_ll(q, BASE, "scene", "train")
    out[name] = {s: boot(q, fold_ll(q, BASE + [s], "scene", "train") - lq, "scene") for s in SIGS}
# descriptive corrections
B = analyze.B
cnt = {}
for _sc, ev in B.items():
    for e, locs in ev.items():
        cnt.setdefault(e, [0, 0])
        cnt[e][0] += 1
        cnt[e][1] += len(locs)
out["boundaries_by_type_videos_bins"] = cnt
rows = list(csv.DictReader(open(EVENT_TRIALS)))
catch = [
    ast.literal_eval(r["Trial_Info"])
    for r in rows
    if "non-visible" in r["Trial_Info"] and "TrialName" in r["Trial_Info"]
]
catch = [t for t in catch if isinstance(t, dict) and not t.get("IsInstruction")]
out["catch_raw_keypress_rate"] = [sum(len(t["Spacebar"]) > 0 for t in catch), len(catch)]
s = tr[tr.probe].groupby(["sub", "boundary"]).detected.mean().unstack().dropna()
dd = s[1] - s[0]
out["paired_n"] = int(len(dd))
out["dz_sample_sd"] = float(dd.mean() / dd.std(ddof=1))
out["n_bins_extracted_analyzed"] = [int(len(pd.read_csv(RESULTS / "events_features.csv"))), int(len(feat))]
out["n_human_bins"] = int(p0.groupby(["scene", "bin"]).ngroups)
# raw Woven agreement (no smoothing)
for task, f in [("stiffness", "parsed_questions_stiff.csv"), ("mass", "parsed_questions_mass.csv")]:
    d = pd.read_csv(WOVEN / f, usecols=["subject", "trial", "human_res", "woven_res", "dnn_res"])
    d["aw"] = (d.woven_res == d.human_res).astype(float)
    d["ad"] = (d.dnn_res == d.human_res).astype(float)
    out[f"woven_raw_agreement_{task}"] = dict(
        woven=float(d.aw.mean()), dnn=float(d.ad.mean()), diff=float(d.aw.mean() - d.ad.mean())
    )
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS / "events_sensitivity.json", "w"), indent=1)

# Paired comparison of readouts: DINOv2-change model vs V-JEPA-surprise model (same base), strict inference
lv = fold_ll(p0, BASE + ["vjepa_surprise"], "scene", "train")
ld = fold_ll(p0, BASE + ["dino_change"], "scene", "train")
out2 = {
    "S1_dino_minus_vjepa_scene_boot": boot(p0, ld - lv, "scene"),
    "S1_dino_minus_vjepa_crossed": crossed(p0, ld - lv),
}
lvf = fold_ll(p0, BASE + ["vjepa_surprise"], "family", "train")
ldf = fold_ll(p0, BASE + ["dino_change"], "family", "train")
out2["S2_dino_minus_vjepa_family_boot"] = boot(p0, ldf - lvf, "family")
print(json.dumps(out2, indent=1))
out.update(out2)
json.dump(out, open(RESULTS / "events_sensitivity.json", "w"), indent=1)
