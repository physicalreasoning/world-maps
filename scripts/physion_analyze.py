"""Physion: do imagined outcomes from video generators predict what people predict?

Readouts, all giving a P(yes) per stimulus for "will the red object touch the yellow area":
  humans       P(yes) from the released per-stimulus accuracy and the ground-truth label (66-96 participants each)
  kinematic    logistic readout on hand-made context features (distance, approach speed, offsets): the appearance /
               extrapolation heuristic
  encoders     logistic readout on frozen video/image features of the context (within-scenario CV and
               leave-one-scenario-out)
  vlm          Qwen3-VL yes/no probability
  generator    the contact detector applied to video continuations (mean over samples), with a per-scenario
               calibration of the detector fitted on REAL outcome frames (measurement, not on human data)

Metrics: accuracy vs ground truth; agreement with humans (Pearson r with human P(yes), Cohen's kappa with the human
majority); error-pattern agreement = partial correlation after regressing the label out of both (does the model get
the same stimuli wrong?); accuracy-matched null agents; stimulus bootstraps; paired comparisons on the generator subset.
Detector validity gate: scenarios where the detector reaches >= 0.75 accuracy on real outcomes form the primary set.
Outputs: results/physion/physion_results.json / .md, figures/*.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from world_maps import physion as P  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "physion"
CS = [0.01, 0.1, 1.0, 10.0]


def logit_cv(X, y, groups=None, seed=0, n_splits=5):
    """Cross-validated logistic P(yes). groups=None -> stratified K-fold; else leave-one-group-out."""
    X = np.nan_to_num(np.asarray(X, float))
    y = np.asarray(y).astype(int)
    p = np.full(len(y), np.nan)
    if groups is None:
        folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    else:
        folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        sc = StandardScaler().fit(X[tr])
        best, bestc = -1, 1.0
        for C in CS:
            inner = StratifiedKFold(3, shuffle=True, random_state=seed)
            accs = []
            for itr, ite in inner.split(X[tr], y[tr]):
                m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr][itr]), y[tr][itr])
                accs.append((m.predict(sc.transform(X[tr][ite])) == y[tr][ite]).mean())
            if np.mean(accs) > best:
                best, bestc = np.mean(accs), C
        m = LogisticRegression(C=bestc, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def agreement(p_model, p_human, label, n_boot=1000, seed=0):
    """Accuracy, r with human P(yes), kappa with the human majority, partial r given the label, with stimulus bootstraps."""
    p_model, p_human, label = np.asarray(p_model, float), np.asarray(p_human, float), np.asarray(label).astype(int)
    ok = ~np.isnan(p_model) & ~np.isnan(p_human)
    p_model, p_human, label = p_model[ok], p_human[ok], label[ok]
    if len(label) < 8:
        return {"n": int(len(label))}

    def partial_r(a, b, c):
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")

    def m(idx):
        pm, ph, lb = p_model[idx], p_human[idx], label[idx]
        r = float(np.corrcoef(pm, ph)[0, 1]) if pm.std() > 0 and ph.std() > 0 else float("nan")
        return {"accuracy": float(((pm > 0.5).astype(int) == lb).mean()), "r_human": r,
                "kappa_majority": float(cohen_kappa_score((pm > 0.5).astype(int), (ph > 0.5).astype(int))),
                "partial_r_given_label": partial_r(pm, ph, lb.astype(float)),
                "auc": float(roc_auc_score(lb, pm)) if len(np.unique(lb)) > 1 else float("nan")}

    point = m(np.arange(len(label)))
    rng = np.random.default_rng(seed)
    boots = [m(rng.integers(0, len(label), len(label))) for _ in range(n_boot)]
    out = {"n": int(len(label))}
    for k, v in point.items():
        bs = np.array([b[k] for b in boots], float)
        out[k] = v
        out[f"{k}_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    # accuracy-matched null: random agents with the same accuracy and independent errors
    acc = point["accuracy"]
    null_r, null_k, null_pr = [], [], []
    for _ in range(200):
        correct = rng.random(len(label)) < acc
        pm = np.where(correct, label, 1 - label).astype(float)
        pm = np.clip(pm + rng.normal(0, 0.05, len(pm)), 0, 1)
        null_r.append(np.corrcoef(pm, p_human)[0, 1] if pm.std() > 0 else np.nan)
        null_k.append(cohen_kappa_score((pm > 0.5).astype(int), (p_human > 0.5).astype(int)))
        null_pr.append(partial_r(pm, p_human, label.astype(float)))
    out["null_r_human"], out["null_kappa"], out["null_partial_r"] = float(np.nanmean(null_r)), float(np.nanmean(null_k)), float(np.nanmean(null_pr))
    out["excess_partial_r"] = float(point["partial_r_given_label"] - np.nanmean(null_pr))
    out["_boot_partial"] = np.array([b["partial_r_given_label"] for b in boots], float)
    out["_boot_r"] = np.array([b["r_human"] for b in boots], float)
    return out


def paired_diff(a_boot, b_boot):
    d = a_boot - b_boot
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    return {"diff_mean": float(d.mean()), "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def human_reliability(df):
    """Reliability of the per-stimulus human P(yes): 1 - mean binomial sampling variance / observed variance."""
    out = {}
    for scen, g in df.groupby("scenario"):
        p, n = g.p_yes_human.values, g.n_human.values
        sv = np.mean(p * (1 - p) / n)
        out[scen] = float(1 - sv / p.var()) if p.var() > 0 else float("nan")
    p, n = df.p_yes_human.values, df.n_human.values
    out["all"] = float(1 - np.mean(p * (1 - p) / n) / p.var())
    return out


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--gate", type=float, default=0.75)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    stim = pd.read_csv(out / "stimuli.csv")
    stim["label"] = stim.label.astype(str).str.lower().eq("true")
    stim = stim.dropna(subset=["p_yes_human"]).reset_index(drop=True)
    res = {"n_stimuli": int(len(stim)), "human": {}, "detector": {}, "readouts": {}, "generator": {}, "comparisons": {}}

    # ---- humans
    res["human"]["accuracy_by_scenario"] = {k: float(v) for k, v in stim.groupby("scenario").human_acc.mean().items()}
    res["human"]["p_yes_mean"] = float(stim.p_yes_human.mean())
    res["human"]["reliability"] = human_reliability(stim)

    # ---- detector validation and calibration on REAL outcomes
    det = pd.read_csv(out / "detector_real.csv") if (out / "detector_real.csv").exists() else None
    calib = {}
    gate_ok = set(P.SCENARIOS)
    if det is not None:
        det["label"] = det.label.astype(str).str.lower().eq("true")
        det["real_contact"] = det.real_contact.map(lambda v: np.nan if pd.isna(v) else str(v).lower() == "true")
        by = {}
        for scen, g in det.groupby("scenario"):
            ok = g.real_contact.notna()
            acc = float((g.real_contact[ok] == g.label[ok]).mean()) if ok.any() else float("nan")
            feats = g[["real_score", "real_max_overlap", "real_min_dist", "real_base_overlap"]].copy()
            feats["real_min_dist"] = feats.real_min_dist.clip(upper=60)
            feats["vanish"] = g.real_vanish.astype(str).str.lower().eq("true").astype(float)
            pcal = logit_cv(feats.values, g.label.values, seed=0)
            by[scen] = {"rule_accuracy": acc, "rule_auc": float(roc_auc_score(g.label, g.real_score.fillna(0))),
                        "calibrated_cv_accuracy": float(((pcal > 0.5) == g.label.values).mean()), "n": int(len(g)),
                        "frac_valid": float(g.real_frac_valid.mean())}
            sc = StandardScaler().fit(np.nan_to_num(feats.values))
            calib[scen] = (sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(np.nan_to_num(feats.values)), g.label.values.astype(int)))
        res["detector"]["by_scenario"] = by
        gate_ok = {s for s, v in by.items() if max(v["rule_accuracy"], v["calibrated_cv_accuracy"]) >= args.gate}
        res["detector"]["primary_scenarios"] = sorted(gate_ok)
        res["detector"]["overall_rule_accuracy"] = float((det.real_contact == det.label)[det.real_contact.notna()].mean())

    # ---- readouts on the full set
    preds = {}
    kin = pd.read_csv(out / "kinematics.csv") if (out / "kinematics.csv").exists() else None
    if kin is not None:
        k = stim[["stim_id", "scenario", "label"]].merge(kin, on="stim_id", how="left")
        X = k[[c for c in k.columns if c.startswith("k")]].values
        preds["kinematic/within"] = pd.Series(logit_cv(X, k.label.values), index=k.stim_id)
        preds["kinematic/loso"] = pd.Series(logit_cv(X, k.label.values, groups=k.scenario.values), index=k.stim_id)
    for f in sorted(glob.glob(str(out / "features" / "*.npz"))):
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        ids = z["stim_id"]
        order = pd.Series(np.arange(len(ids)), index=ids).reindex(stim.stim_id).values
        okrow = ~pd.isna(order)
        for key in [k for k in z.files if k.startswith("X_")]:
            Xf = z[key].astype(np.float32)[order[okrow].astype(int)]
            Xp = PCA(min(128, Xf.shape[0] - 1, Xf.shape[1]), random_state=0).fit_transform(np.nan_to_num(Xf))
            lab = stim.label.values[okrow]
            s_w = pd.Series(np.nan, index=stim.stim_id)
            s_w.iloc[np.where(okrow)[0]] = logit_cv(Xp, lab)
            s_l = pd.Series(np.nan, index=stim.stim_id)
            s_l.iloc[np.where(okrow)[0]] = logit_cv(Xp, lab, groups=stim.scenario.values[okrow])
            preds[f"{name}/{key[2:]}/within"] = s_w
            preds[f"{name}/{key[2:]}/loso"] = s_l
    if (out / "vlm.csv").exists():
        v = pd.read_csv(out / "vlm.csv").drop_duplicates("stim_id").set_index("stim_id").p_yes
        preds["vlm"] = v.reindex(stim.stim_id)

    def evaluate(series, subset=None):
        d = stim.set_index("stim_id")
        idx = series.index if subset is None else [i for i in series.index if i in subset]
        d = d.loc[idx]
        pm = series.loc[idx].values
        allres = {"all": agreement(pm, d.p_yes_human.values, d.label.values, n_boot=args.n_boot)}
        prim = [i for i in idx if d.loc[i, "scenario"] in gate_ok]
        allres["primary"] = agreement(series.loc[prim].values, d.loc[prim].p_yes_human.values, d.loc[prim].label.values, n_boot=args.n_boot)
        allres["by_scenario"] = {scen: strip(agreement(series.loc[g.index].values, g.p_yes_human.values, g.label.values, n_boot=100))
                                 for scen, g in d.groupby("scenario")}
        return allres

    for name, s in preds.items():
        res["readouts"][name] = {k: (strip(v) if isinstance(v, dict) and "_boot_partial" in v else v) for k, v in evaluate(s).items()}

    # ---- generators
    gen_boot = {}
    for sf in sorted(glob.glob(str(out / "generated" / "*" / "scores.csv"))):
        gname = Path(sf).parent.name
        g = pd.read_csv(sf)
        g["label"] = g.label.astype(str).str.lower().eq("true")
        feats = g[["gen_score", "gen_max_overlap", "gen_min_dist", "gen_base_overlap"]].copy()
        feats["gen_min_dist"] = feats.gen_min_dist.clip(upper=60)
        feats["vanish"] = g.gen_vanish.astype(str).str.lower().eq("true").astype(float)
        pcal = np.full(len(g), np.nan)
        for scen in g.scenario.unique():
            sel = (g.scenario == scen).values
            if scen in calib:
                sc, m = calib[scen]
                pcal[sel] = m.predict_proba(sc.transform(np.nan_to_num(feats.values[sel])))[:, 1]
            else:
                pcal[sel] = g.gen_contact[sel].astype(str).str.lower().eq("true").astype(float)
        g["p_cal"] = pcal
        g["p_rule"] = g.gen_contact.astype(str).str.lower().eq("true").astype(float)
        per = g.groupby("stim_id").agg(p_cal=("p_cal", "mean"), p_rule=("p_rule", "mean"), k=("sample", "size"))
        subset = set(per.index)
        entry = {"n_stimuli": int(len(per)), "samples_per_stimulus": float(per.k.mean()), "n_failed_detection": int(g.gen_contact.isna().sum())}
        for key in ["p_cal", "p_rule"]:
            s = per[key].reindex([i for i in stim.stim_id if i in subset])
            ev = evaluate(s, subset=subset)
            gen_boot[(gname, key)] = ev
            entry[key] = {k: (strip(v) if isinstance(v, dict) and "_boot_partial" in v else v) for k, v in ev.items()}
        # the other readouts on the same subset, for paired comparisons
        comp = {}
        for name, s in preds.items():
            ev = evaluate(s, subset=subset)
            comp[name] = {k: (strip(v) if isinstance(v, dict) and "_boot_partial" in v else v) for k, v in ev.items()}
            for level in ["all", "primary"]:
                a, b = gen_boot[(gname, "p_cal")][level], ev[level]
                if "_boot_partial" in a and "_boot_partial" in b:
                    res["comparisons"][f"{gname} vs {name} [{level}]"] = {
                        "partial_r": paired_diff(a["_boot_partial"], b["_boot_partial"]), "r_human": paired_diff(a["_boot_r"], b["_boot_r"])}
        entry["others_on_subset"] = comp
        res["generator"][gname] = entry

    # ---- scenario profile correlation with humans (accuracy by scenario)
    hum_prof = pd.Series(res["human"]["accuracy_by_scenario"])
    res["scenario_profile_r"] = {}
    for name, r in res["readouts"].items():
        prof = pd.Series({k: v.get("accuracy", np.nan) for k, v in r["by_scenario"].items()}).reindex(hum_prof.index)
        if prof.notna().sum() >= 5:
            res["scenario_profile_r"][name] = float(stats.spearmanr(prof.values, hum_prof.values, nan_policy="omit").correlation)

    figures(res, out / "figures")
    with open(out / "physion_results.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else (o.item() if hasattr(o, "item") else str(o)))
    md = summarize(res)
    (out / "physion_results.md").write_text(md)
    print(md)


def summarize(res):
    f = lambda v, d=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    L = ["# Physion: generators as simulators vs readouts, against human predictions", ""]
    L.append(f"Stimuli with human data: {res['n_stimuli']}. Human P(yes) reliability (binomial-corrected): all {f(res['human']['reliability'].get('all'))}; "
             f"human accuracy by scenario: " + ", ".join(f"{k} {v:.2f}" for k, v in res["human"]["accuracy_by_scenario"].items()))
    if res["detector"]:
        L.append("Detector on real outcomes (rule / calibrated CV accuracy): " + ", ".join(
            f"{k} {v['rule_accuracy']:.2f}/{v['calibrated_cv_accuracy']:.2f}" for k, v in res["detector"]["by_scenario"].items())
                 + f". Primary scenarios (>= gate): {', '.join(res['detector']['primary_scenarios'])}.")
    L += ["", "## Readouts on all stimuli", "", "| readout | n | accuracy | r with human P(yes) [CI] | κ vs majority | partial r given label [CI] (null) | primary: partial r [CI] |", "|---|---|---|---|---|---|---|"]
    for name, r in res["readouts"].items():
        a, p = r["all"], r["primary"]
        L.append(f"| {name} | {a.get('n')} | {f(a.get('accuracy'))} | {f(a.get('r_human'))} [{f(a.get('r_human_ci',[None,None])[0])}, {f(a.get('r_human_ci',[None,None])[1])}] | "
                 f"{f(a.get('kappa_majority'))} | {f(a.get('partial_r_given_label'))} [{f(a.get('partial_r_given_label_ci',[None,None])[0])}, {f(a.get('partial_r_given_label_ci',[None,None])[1])}] ({f(a.get('null_partial_r'))}) | "
                 f"{f(p.get('partial_r_given_label'))} [{f(p.get('partial_r_given_label_ci',[None,None])[0])}, {f(p.get('partial_r_given_label_ci',[None,None])[1])}] |")
    for gname, g in res["generator"].items():
        L += ["", f"## Generator: {gname} ({g['n_stimuli']} stimuli, {g['samples_per_stimulus']:.1f} samples each)", "",
              "| readout (same subset) | accuracy | r with human P(yes) [CI] | κ | partial r given label [CI] (null) | primary: partial r [CI] |", "|---|---|---|---|---|---|"]
        for key in ["p_cal", "p_rule"]:
            a, p = g[key]["all"], g[key]["primary"]
            L.append(f"| **{gname} {key}** | {f(a.get('accuracy'))} | {f(a.get('r_human'))} [{f(a.get('r_human_ci',[None,None])[0])}, {f(a.get('r_human_ci',[None,None])[1])}] | {f(a.get('kappa_majority'))} | "
                     f"{f(a.get('partial_r_given_label'))} [{f(a.get('partial_r_given_label_ci',[None,None])[0])}, {f(a.get('partial_r_given_label_ci',[None,None])[1])}] ({f(a.get('null_partial_r'))}) | "
                     f"{f(p.get('partial_r_given_label'))} [{f(p.get('partial_r_given_label_ci',[None,None])[0])}, {f(p.get('partial_r_given_label_ci',[None,None])[1])}] |")
        for name, r in g["others_on_subset"].items():
            a, p = r["all"], r["primary"]
            L.append(f"| {name} | {f(a.get('accuracy'))} | {f(a.get('r_human'))} [{f(a.get('r_human_ci',[None,None])[0])}, {f(a.get('r_human_ci',[None,None])[1])}] | {f(a.get('kappa_majority'))} | "
                     f"{f(a.get('partial_r_given_label'))} [{f(a.get('partial_r_given_label_ci',[None,None])[0])}, {f(a.get('partial_r_given_label_ci',[None,None])[1])}] ({f(a.get('null_partial_r'))}) | "
                     f"{f(p.get('partial_r_given_label'))} [{f(p.get('partial_r_given_label_ci',[None,None])[0])}, {f(p.get('partial_r_given_label_ci',[None,None])[1])}] |")
        L += ["", "Paired comparisons (generator calibrated P(yes) minus readout; stimulus bootstrap):"]
        for k, v in res["comparisons"].items():
            if k.startswith(gname) and v.get("partial_r"):
                L.append(f"- {k}: partial r diff {v['partial_r']['diff_mean']:+.3f} [{v['partial_r']['diff_ci'][0]:+.3f}, {v['partial_r']['diff_ci'][1]:+.3f}] (p = {v['partial_r']['p_two_sided']:.3f}); "
                         f"r diff {v['r_human']['diff_mean']:+.3f} [{v['r_human']['diff_ci'][0]:+.3f}, {v['r_human']['diff_ci'][1]:+.3f}]")
    if res.get("scenario_profile_r"):
        L += ["", "Scenario-profile correlation with human accuracy (Spearman over 8 scenarios): " + ", ".join(f"{k} {v:.2f}" for k, v in res["scenario_profile_r"].items())]
    L += ["", "partial r given label: correlation between model and human P(yes) after the ground-truth label is regressed out of both, i.e. agreement on *which* stimuli are judged wrongly. "
          "null = accuracy-matched random agent. Primary = scenarios where the contact detector passed the validity gate on real outcomes."]
    return "\n".join(L)


def figures(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name, r in res["readouts"].items():
        rows.append((name, r["all"].get("partial_r_given_label"), r["all"].get("partial_r_given_label_ci", [np.nan, np.nan]), r["all"].get("accuracy"), "#36c"))
    for gname, g in res["generator"].items():
        a = g["p_cal"]["all"]
        rows.append((f"{gname} (imagined outcome)", a.get("partial_r_given_label"), a.get("partial_r_given_label_ci", [np.nan, np.nan]), a.get("accuracy"), "#2a2"))
    rows = [r for r in rows if r[1] is not None and np.isfinite(r[1])]
    if not rows:
        return
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(rows) + 3), 4.5))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[[r[1] - r[2][0] for r in rows], [r[2][1] - r[1] for r in rows]], color=[r[4] for r in rows])
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\nacc {r[3]:.2f}" for r in rows], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("partial r with human P(yes) | label"); ax.set_title("Which readout gets the same stimuli wrong as people?")
    fig.tight_layout(); fig.savefig(out / "error_agreement.png", dpi=150); plt.close(fig)
    # scenario heatmap of accuracy
    names = list(res["readouts"]) + [f"{g} (imagined)" for g in res["generator"]]
    scen = list(res["human"]["accuracy_by_scenario"])
    M = np.full((len(names) + 1, len(scen)), np.nan)
    M[0] = [res["human"]["accuracy_by_scenario"][s] for s in scen]
    for i, n in enumerate(list(res["readouts"])):
        M[i + 1] = [res["readouts"][n]["by_scenario"].get(s, {}).get("accuracy", np.nan) for s in scen]
    for j, g in enumerate(res["generator"]):
        M[len(res["readouts"]) + 1 + j] = [res["generator"][g]["p_cal"]["by_scenario"].get(s, {}).get("accuracy", np.nan) for s in scen]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(names) + 2))
    im = ax.imshow(M, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(names) + 1)); ax.set_yticklabels(["humans"] + names, fontsize=6)
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen, rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=ax, label="accuracy"); fig.tight_layout(); fig.savefig(out / "accuracy_by_scenario.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
