"""Pilot 0 analysis (pre-specified before any model signal was compared with human data
or boundaries; only one scene's feature distributions had been inspected).

A. Reproduce the published human effect with the lab's own exclusion rules
   (Exp1_TemporalProbe.ipynb): slowdowns at expert event boundaries are detected less often.
B. Boundary localization: within-scene AUROC of each signal for expert-boundary bins vs all
   other bins; mean over scenes; 95% scene-bootstrap CI; paired difference vs flow_accel.
C. Human prediction: trial-level logistic regression, leave-one-scene-out. Base model =
   low-level covariates (lab pixel change of the probe video, pix_diff, flow_speed,
   flow_accel, time in video, time^2). Test model = base + one signal (z-scored within scene).
   Statistic: held-out log-likelihood gain per trial (x1000), scene-bootstrap CI, Holm over the
   five model signals. Primary = vjepa_surprise.
E. (exploratory, added after cycle-1 review, before results) lag profile: within-scene partial
   Spearman correlation between per-bin human detection rate at bin k and signal at bin k+L,
   L in -3..3, residualizing both on the low-level covariates; scene-bootstrap CI.
D. Boundary-effect attenuation: scene fixed-effects logistic regression of detection on
   boundary + base covariates, with and without the signal; % reduction in the boundary
   coefficient, scene-bootstrap CI.
"""

import ast
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from world_maps.core import holm_adjust
from world_maps.paths import BOUNDARIES, EVENT_TRIALS, PIXEL_CHANGES, RESULTS

rng = np.random.default_rng(2026)
B = json.load(open(BOUNDARIES))
PIX = str(PIXEL_CHANGES / "{}_pixel_changes.json")
MODEL = ["vjepa_surprise", "vjepa_surprise_top5", "vjepa_change", "vmae_surprise", "dino_change"]
LOW = ["pix_diff", "flow_speed", "flow_accel"]
NBOOT = 2000


def bounds(scene):
    return set(int(x) for v in B[scene].values() for x in v)


# ---------------- A. human data, lab rules ----------------
def load_trials():
    rows = list(csv.DictReader(open(EVENT_TRIALS)))
    by = defaultdict(list)
    for r in rows:
        try:
            t = ast.literal_eval(r["Trial_Info"])
        except Exception:
            continue
        if isinstance(t, dict) and "TrialName" in t and not t.get("IsInstruction", False):
            by[r["Participant_ID"]].append(t)
    pixcache = {}
    out, excluded = [], []
    for sub, trials in by.items():
        ignored, keep = 0, []
        for tr in trials:
            name = tr["TrialName"]
            k = int(name.split("_")[-1].split(".")[0])
            sp = tr["Spacebar"]
            ignore = False
            if len(sp) == 0:
                st, rt = np.nan, np.nan
            elif len(sp) > 1:
                st, rt, ignore = np.nan, np.nan, True
            else:
                st = sp[0]
                rt = (st - 3000) - k * 200
            probe = "non-visible" not in name
            if rt > 2500 or rt < 300:
                ignore = True
            if ignore:
                ignored += 1
                continue
            scene = name.split("/")[0]
            if probe:
                if scene not in pixcache:
                    pixcache[scene] = json.load(open(PIX.format(scene)))
                pc = pixcache[scene].get(name, np.nan)
            else:
                pc = np.nan
            keep.append(
                dict(
                    sub=sub,
                    scene=scene,
                    bin=k,
                    probe=probe,
                    detected=int(not np.isnan(st)),
                    boundary=int(k in bounds(scene)),
                    lab_pix=pc,
                )
            )
        if ignored >= 48:
            excluded.append(sub)
            continue
        out += keep
    return pd.DataFrame(out), excluded


def reproduce(tr):
    p = tr[tr.probe]
    s = p.groupby(["sub", "boundary"]).detected.mean().unstack()
    t = stats.ttest_rel(s[1], s[0], nan_policy="omit")
    d = (s[1] - s[0]).mean() / (s[1] - s[0]).std(ddof=0)
    return dict(
        n_participants=int(s.shape[0]),
        boundary=float(s[1].mean()),
        nonboundary=float(s[0].mean()),
        t=float(t.statistic),
        p=float(t.pvalue),
        cohen_dz=float(d),
        n_probe_trials=int(len(p)),
        false_alarm_rate_catch=float(tr[~tr.probe].detected.mean()),
    )


# ---------------- B. boundary localization ----------------
def boundary_auc(feat, sig):
    per = {}
    for sc, g in feat.groupby("scene"):
        y = g.bin.isin(bounds(sc)).astype(int).values
        x = g[sig].values
        if 0 < y.sum() < len(y) and np.isfinite(x).all():
            per[sc] = roc_auc_score(y, x)
    return per


def boot_mean(per_scene_vals, n=NBOOT):
    v = np.array(list(per_scene_vals))
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------- C. held-out human prediction ----------------
def zwithin(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df.groupby("scene")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return df


def loso_ll(df, cols):
    """per-trial held-out log-likelihood, leave-one-scene-out."""
    ll = np.zeros(len(df))
    for sc in df.scene.unique():
        te = (df.scene == sc).values
        m = LogisticRegression(C=1.0, max_iter=2000).fit(
            df.loc[~te, cols].values, df.loc[~te, "detected"].values
        )
        p = np.clip(m.predict_proba(df.loc[te, cols].values)[:, 1], 1e-6, 1 - 1e-6)
        y = df.loc[te, "detected"].values
        ll[te] = y * np.log(p) + (1 - y) * np.log(1 - p)
    return ll


def scene_boot_diff(df, a, b, n=NBOOT):
    d = pd.Series(b - a).groupby(df.scene.values).agg(["sum", "count"])
    scenes = d.index.values
    est = d["sum"].sum() / d["count"].sum()
    bs = []
    for _ in range(n):
        s = rng.choice(scenes, len(scenes))
        bs.append(d.loc[s, "sum"].sum() / d.loc[s, "count"].sum())
    pval = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(max(pval, 1 / n))


# ---------------- D. boundary attenuation ----------------
def boundary_coef(df, cols):
    X = pd.get_dummies(df.scene, drop_first=True, dtype=float)
    X = pd.concat([df[["boundary"] + cols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    m = LogisticRegression(C=1e4, max_iter=5000).fit(X.values, df.detected.values)
    return m.coef_[0][0]


def main(feat_path):
    tr, excluded = load_trials()
    res = {"A_reproduction": reproduce(tr), "excluded_participants": len(excluded)}
    feat = (
        pd.read_csv(feat_path).dropna().reset_index(drop=True)
    )  # drop final bins whose window passes the clip end
    sigs = MODEL + LOW
    # B
    resB = {}
    ref = boundary_auc(feat, "flow_accel")
    for s in sigs + ["vjepa_mismatch"]:
        per = boundary_auc(feat, s)
        m, lo, hi = boot_mean(per.values())
        common = [k for k in per if k in ref]
        diff = boot_mean([per[k] - ref[k] for k in common])
        resB[s] = dict(auc=m, lo=lo, hi=hi, minus_flow_accel=diff)
    res["B_boundary_auc"] = resB
    # sanity: true future predicted better than another scene's future
    res["sanity_surprise_lt_mismatch"] = float((feat.vjepa_surprise < feat.vjepa_mismatch).mean())
    # C
    p = tr[tr.probe].merge(feat, on=["scene", "bin"], how="inner")
    p["t"] = p.bin * 0.2
    p["t2"] = p.t**2
    p["lab_pix"] = p.lab_pix.fillna(p.lab_pix.median())
    p = zwithin(p, sigs + ["lab_pix"])
    p[["t", "t2"]] = (p[["t", "t2"]] - p[["t", "t2"]].mean()) / p[["t", "t2"]].std()
    base = ["lab_pix", "pix_diff", "flow_speed", "flow_accel", "t", "t2"]
    ll0 = loso_ll(p, base)
    resC = {"n_trials": int(len(p)), "n_scenes": int(p.scene.nunique())}
    pv = {}
    for s in MODEL + ["boundary"]:
        ll1 = loso_ll(p, base + [s])
        est, lo, hi, pval = scene_boot_diff(p, ll0, ll1)
        resC[s] = dict(dLL_per_trial_x1000=1000 * est, lo=1000 * lo, hi=1000 * hi, p=pval)
        if s in MODEL:
            pv[s] = pval
    for s, adj in holm_adjust(pv).items():
        resC[s]["p_holm"] = adj
    res["C_heldout_human"] = resC
    # D
    b0 = boundary_coef(p, base)
    resD = {"boundary_coef_base": float(b0)}
    scenes = p.scene.unique()
    for s in MODEL:
        b1 = boundary_coef(p, base + [s])
        bs = []
        for _ in range(200):
            ss = rng.choice(scenes, len(scenes))
            q = pd.concat([p[p.scene == x].assign(scene=f"{x}_{i}") for i, x in enumerate(ss)])
            c0, c1 = boundary_coef(q, base), boundary_coef(q, base + [s])
            bs.append(1 - c1 / c0)
        resD[s] = dict(
            pct_reduction=float(100 * (1 - b1 / b0)),
            lo=float(100 * np.percentile(bs, 2.5)),
            hi=float(100 * np.percentile(bs, 97.5)),
        )
    res["D_boundary_attenuation"] = resD
    # E. lag profile on per-bin rates
    rate = (
        tr[tr.probe]
        .groupby(["scene", "bin"])
        .agg(det=("detected", "mean"), n=("detected", "size"), lab_pix=("lab_pix", "mean"))
        .reset_index()
    )
    resE = {}
    for s in ["vjepa_surprise", "vjepa_surprise_top5", "flow_accel"]:
        prof = {}
        for L in range(-3, 4):
            f2 = feat[["scene", "bin", s] + LOW].copy()
            f2["bin"] = f2["bin"] - L  # signal at k+L aligned to bin k
            f2 = f2.rename(columns={s: "sig"} if s not in LOW else {})
            if s in LOW:
                f2["sig"] = feat[s].values
            q = rate.merge(f2[["scene", "bin", "sig"]], on=["scene", "bin"]).merge(
                feat[["scene", "bin"] + LOW], on=["scene", "bin"]
            )
            per = []
            for _sc, g in q.groupby("scene"):
                if len(g) < 10:
                    continue
                Z = np.column_stack([np.ones(len(g))] + [stats.rankdata(g[c]) for c in LOW + ["bin"]])
                ry = stats.rankdata(g.det)
                rx = stats.rankdata(g.sig)
                ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
                ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
                if ex.std() > 0 and ey.std() > 0:
                    per.append(np.corrcoef(ex, ey)[0, 1])
            m, lo, hi = boot_mean(per)
            prof[L] = dict(r=m, lo=lo, hi=hi)
        resE[s] = prof
    res["E_lag_profile_exploratory"] = resE
    print(json.dumps(res, indent=1))
    json.dump(res, open(RESULTS / "events_results.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RESULTS / "events_features.csv"))
