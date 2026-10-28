"""Shape under cloth: humans, PbAS and modern models on the same trials.

Readouts
  BU (paper protocol): d1 = corr(e_target, e_match), d2 = corr(e_target, e_distractor); choice = d1 > d2;
     continuous score = Luce rule on (1 + corr) / 2; margin = |d1 - d2| as a confidence proxy for RT.
  De-draping (new, D-style): ridge map from the draped target's embedding to the bare object's embedding,
     learned on the other trials (leave-one-trial-out) or on the other categories (leave-category-out),
     then the same 2AFC on the mapped embedding. Tests whether the hidden shape is linearly recoverable
     from the features even when raw similarity does not use it.
  Imagination (new): generator-in-the-loop analysis-by-synthesis. 'drape': imagined draped versions of each
     candidate compared with the target; 'undrape': imagined bare objects from the target compared with the
     candidates. Sequential sampling over imagined samples gives an RT prediction (samples to criterion),
     as PbAS's iterations do.
  VLM: Qwen3-VL 2AFC probability.

Human side follows the lab's scripts (exclusions, per-trial accuracy, RT z-filter, participant bootstrap,
split-half ceiling). PbAS and Pixel-PbAS per-trial predictions come from the released run arrays.
Outputs: results/cloth/cloth_results.json / .md, figures/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from world_maps import cloth  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
ALPHAS = np.logspace(-2, 6, 17)
SAME = np.array([((t // 6) % 2 == 0) for t in range(120)])


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between matched rows of a and b."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def luce(d1, d2):
    p1, p2 = (1 + d1) / 2, (1 + d2) / 2
    return p1 / (p1 + p2 + 1e-9)


def boot_corr(model: np.ndarray, human_boot: np.ndarray, sel=None) -> dict:
    """Correlation between a per-trial model score and participant-bootstrapped human accuracy."""
    sel = np.arange(120) if sel is None else np.asarray(sel)
    m = model[sel]
    if m.std() == 0:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")], "point": float("nan")}
    rs = np.array([np.corrcoef(m, hb[sel])[0, 1] if hb[sel].std() > 0 else np.nan for hb in human_boot])
    point = float(np.corrcoef(m, human_boot.mean(0)[sel])[0, 1])
    return {"mean": float(np.nanmean(rs)), "ci": [float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))], "point": point, "_rs": rs}


def compare_boot(rs_a: np.ndarray, rs_b: np.ndarray) -> dict:
    d = rs_a - rs_b
    d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff_mean": float(d.mean()), "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_two_sided": float(min(1.0, p))}


# ----------------------------------------------------------------------------- features

class Feats:
    def __init__(self, path: Path, index: pd.DataFrame):
        z = np.load(path)
        self.layers = [k[2:] for k in z.files if k.startswith("X_")]
        self.X = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("X_")}
        self.index = index.reset_index(drop=True)
        self.lookup = {}
        for i, r in self.index.iterrows():
            self.lookup[(r.condition, int(r.trial), r.role, r.source, int(r["sample"]))] = i

    def get(self, layer, condition, trial, role, source="stim", sample=0):
        i = self.lookup.get((condition, trial, role, source, sample))
        return None if i is None else self.X[layer][i]

    def triplets(self, layer, condition):
        T = np.array([self.get(layer, condition, t, "target") for t in range(120)])
        M = np.array([self.get(layer, condition, t, "match") for t in range(120)])
        D = np.array([self.get(layer, condition, t, "distractor") for t in range(120)])
        return T, M, D

    def samples(self, layer, source, role):
        """Imagined samples: dict trial -> array (K, D)."""
        out = {}
        for t in range(120):
            ks = [k for (c, tt, r, s, k) in self.lookup if tt == t and r == role and s == source]
            if ks:
                out[t] = np.array([self.get(layer, "occluded", t, role, source, k) for k in sorted(ks)])
        return out


# ----------------------------------------------------------------------------- readouts

def bu_scores(T, M, D):
    d1, d2 = corr_rows(T, M), corr_rows(T, D)
    return {"d1": d1, "d2": d2, "correct": (d1 > d2).astype(float), "score": luce(d1, d2), "margin": np.abs(d1 - d2)}


def cell_accuracy(correct: np.ndarray) -> dict:
    return {"all": float(correct.mean()), "same_category": float(correct[SAME].mean()), "different_category": float(correct[~SAME].mean())}


def dedrape(T, M, D, mode="loo", k_in=100, k_out=64):
    """Ridge map draped-target -> bare-match embedding, evaluated on held-out trials (loo) or categories (loco).

    Inputs and outputs are compressed with PCA fitted without labels (targets: the 120 draped images; outputs:
    the 240 bare candidates), so the readout is a 100 -> 64 linear map and the 2AFC is scored in the
    64-d bare-object space.
    """
    from sklearn.decomposition import PCA

    p_in = PCA(min(k_in, T.shape[0] - 1, T.shape[1])).fit(T)
    p_out = PCA(min(k_out, 2 * M.shape[0] - 1, M.shape[1])).fit(np.vstack([M, D]))
    Tz, Mz, Dz = p_in.transform(T), p_out.transform(M), p_out.transform(D)
    pred = np.zeros_like(Mz)
    groups = np.arange(120) if mode == "loo" else np.array([t // 12 for t in range(120)])  # 10 categories x 12 trials
    for g in np.unique(groups):
        te = groups == g
        m = RidgeCV(alphas=ALPHAS).fit(Tz[~te], Mz[~te])
        pred[te] = m.predict(Tz[te])
    return bu_scores(pred, Mz, Dz)


def sequential_rt(ev: np.ndarray, target_acc: float, repeats: int = 50, seed: int = 0) -> dict:
    """ev: (120, K) per-sample evidence (match minus distractor). Accumulate in random order until |sum| >= theta.

    theta is chosen so that the mean decision accuracy over trials matches target_acc (grid search).
    Returns per-trial mean number of samples to decision and accuracy.
    """
    rng = np.random.default_rng(seed)
    K = ev.shape[1]
    best = None
    for theta in np.linspace(0.0, np.abs(ev).sum(1).max(), 60):
        acc, nsamp = np.zeros(120), np.zeros(120)
        for r in range(repeats):
            for t in range(120):
                order = rng.permutation(K)
                s = 0.0
                n = K
                for j, k in enumerate(order):
                    s += ev[t, k]
                    if abs(s) >= theta:
                        n = j + 1
                        break
                acc[t] += float(s > 0) if s != 0 else 0.5
                nsamp[t] += n
        acc, nsamp = acc / repeats, nsamp / repeats
        if best is None or abs(acc.mean() - target_acc) < abs(best[0] - target_acc):
            best = (acc.mean(), theta, acc.copy(), nsamp.copy())
    return {"theta": float(best[1]), "mean_acc": float(best[0]), "acc": best[2], "n_samples": best[3]}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(RESULTS / "cloth" / "features"))
    ap.add_argument("--out", default=str(RESULTS / "cloth"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cloth.fetch_assets()

    # ---- humans
    df = cloth.load_behaviour()
    ht = cloth.human_trial_table(df)
    human = {"n_workers_kept": int(df.workerid.nunique()), "cells": {}, "ceiling": {}, "trial_table": ht.to_dict("records")}
    acc = {}
    boot = {}
    for cond in ["unoccluded", "occluded"]:
        for time in ["1s", "2s", "unlimited"]:
            sub = ht[(ht.cond == cond) & (ht.time == time)].set_index("trial").reindex(range(120))
            a = sub.acc.values
            acc[(cond, time)] = a
            human["cells"][f"{cond}/{time}"] = {"all": float(np.nanmean(a)), "same_category": float(np.nanmean(a[SAME])),
                                                "different_category": float(np.nanmean(a[~SAME])), "n_workers": int(df[(df.cond == cond) & (df.time == time)].workerid.nunique()),
                                                "mean_rt_ms": float(np.nanmean(sub.rt.values))}
            boot[(cond, time)] = cloth.participant_bootstrap(df, cond, time, n_boot=args.n_boot)
        human["ceiling"][cond] = {"all": cloth.split_half_ceiling(df, cond, "unlimited"),
                                  "same_category": cloth.split_half_ceiling(df, cond, "unlimited", which=np.where(SAME)[0])}
    log_rt = {cond: ht[(ht.cond == cond) & (ht.time == "unlimited")].set_index("trial").reindex(range(120)).log_rt.values for cond in ["unoccluded", "occluded"]}

    # ---- PbAS reference from the released runs
    ref = {}
    anps = cloth.pbas_arrays()
    for name, anp in anps.items():
        iters = cloth.PBAS_ITER if name == "PbAS" else cloth.PIXEL_PBAS_ITER
        ref[name] = {}
        for cond in ["unoccluded", "occluded"]:
            e = {}
            for time, it in iters.items():
                pa = cloth.pbas_trial_accuracy(anp, cond, it)
                e[time] = {"cells": cell_accuracy(pa), "r_all": {k: v for k, v in boot_corr(pa, boot[(cond, time)]).items() if k != "_rs"},
                           "r_same": {k: v for k, v in boot_corr(pa, boot[(cond, time)], np.where(SAME)[0]).items() if k != "_rs"}}
                if time == "unlimited":
                    e["_rs_all"], e["_rs_same"] = boot_corr(pa, boot[(cond, time)])["_rs"], boot_corr(pa, boot[(cond, time)], np.where(SAME)[0])["_rs"]
            crit = float(np.nanmean(acc[(cond, "unlimited")]))  # the lab's rule: one criterion per occlusion condition
            rt_pred = cloth.pbas_rt_predictor(anp, cond, crit)
            ok = ~np.isnan(log_rt[cond])
            e["rt_r_log"] = float(np.corrcoef(np.log(rt_pred[ok]), log_rt[cond][ok])[0, 1])
            ref[name][cond] = e

    # ---- models
    index = pd.read_csv(feat / "image_index.csv")
    meta = json.load(open(feat / "models.json")) if (feat / "models.json").exists() else {}
    names = [n for n, v in meta.items() if v.get("status") == "ok" and n != "vlm" and (feat / f"{n}.npz").exists()]
    if args.models:
        names = [n for n in names if n in args.models]
    models = {}
    for name in names:
        F = Feats(feat / f"{name}.npz", index)
        models[name] = {"params": meta[name].get("params"), "layers": {}}
        for layer in F.layers:
            entry = {}
            for cond in ["unoccluded", "occluded"]:
                T, M, D = F.triplets(layer, cond)
                bu = bu_scores(T, M, D)
                hb = boot[(cond, "unlimited")]
                r_all, r_same = boot_corr(bu["score"], hb), boot_corr(bu["score"], hb, np.where(SAME)[0])
                ok = ~np.isnan(log_rt[cond])
                e = {"cells": cell_accuracy(bu["correct"]),
                     "r_all": {k: v for k, v in r_all.items() if k != "_rs"}, "r_same": {k: v for k, v in r_same.items() if k != "_rs"},
                     "vs_pbas_same": compare_boot(r_same["_rs"], ref["PbAS"][cond]["_rs_same"]),
                     "margin_vs_log_rt": {"pearson": float(np.corrcoef(bu["margin"][ok], log_rt[cond][ok])[0, 1]),
                                          "spearman": float(stats.spearmanr(bu["margin"][ok], log_rt[cond][ok]).correlation)},
                     "r_all_1s": boot_corr(bu["score"], boot[(cond, "1s")])["mean"], "r_all_2s": boot_corr(bu["score"], boot[(cond, "2s")])["mean"]}
                if cond == "occluded":
                    for mode in ["loo", "loco"]:
                        dd = dedrape(T, M, D, mode)
                        e[f"dedrape_{mode}"] = {"cells": cell_accuracy(dd["correct"]),
                                                "r_all": {k: v for k, v in boot_corr(dd["score"], hb).items() if k != "_rs"},
                                                "r_same": {k: v for k, v in boot_corr(dd["score"], hb, np.where(SAME)[0]).items() if k != "_rs"}}
                    # imagination readouts
                    for src in ["drape", "undrape"]:
                        if src == "drape":
                            SM, SD = F.samples(layer, src, "match"), F.samples(layer, src, "distractor")
                            if len(SM) < 120 or len(SD) < 120:
                                continue
                            K = min(min(v.shape[0] for v in SM.values()), min(v.shape[0] for v in SD.values()))
                            ev = np.array([[corr_rows(T[t:t + 1], SM[t][k:k + 1])[0] - corr_rows(T[t:t + 1], SD[t][k:k + 1])[0] for k in range(K)] for t in range(120)])
                        else:
                            SU = F.samples(layer, src, "target")
                            if len(SU) < 120:
                                continue
                            K = min(v.shape[0] for v in SU.values())
                            ev = np.array([[corr_rows(SU[t][k:k + 1], M[t:t + 1])[0] - corr_rows(SU[t][k:k + 1], D[t:t + 1])[0] for k in range(K)] for t in range(120)])
                        mean_ev = ev.mean(1)
                        correct = (mean_ev > 0).astype(float)
                        score = 1 / (1 + np.exp(-mean_ev / (np.abs(mean_ev).std() + 1e-9)))
                        comb = 0.5 * (stats.zscore(mean_ev) + stats.zscore(bu["d1"] - bu["d2"]))
                        rs = boot_corr(score, hb, np.where(SAME)[0])
                        rt = sequential_rt(ev, np.nanmean(acc[("occluded", "unlimited")]))
                        e[f"imagine_{src}"] = {"K": int(K), "cells": cell_accuracy(correct),
                                               "r_all": {k: v for k, v in boot_corr(score, hb).items() if k != "_rs"},
                                               "r_same": {k: v for k, v in rs.items() if k != "_rs"},
                                               "vs_bu_same": compare_boot(rs["_rs"], r_same["_rs"]),
                                               "vs_pbas_same": compare_boot(rs["_rs"], ref["PbAS"]["occluded"]["_rs_same"]),
                                               "combined_with_bu": {"cells": cell_accuracy((comb > 0).astype(float)),
                                                                    "r_same": {k: v for k, v in boot_corr(comb, hb, np.where(SAME)[0]).items() if k != "_rs"}},
                                               "sequential_rt": {"theta": rt["theta"], "mean_acc": rt["mean_acc"],
                                                                 "r_log_rt": float(np.corrcoef(np.log(rt["n_samples"][ok]), log_rt["occluded"][ok])[0, 1]),
                                                                 "spearman_rt": float(stats.spearmanr(rt["n_samples"][ok], log_rt["occluded"][ok]).correlation)}}
                entry[cond] = e
            models[name]["layers"][layer] = entry
        print(f"{name}: {len(F.layers)} layers", flush=True)

    # ---- VLM
    vlm = {}
    if (feat / "vlm.csv").exists():
        v = pd.read_csv(feat / "vlm.csv")
        for cond in ["unoccluded", "occluded"]:
            p = v[v.condition == cond].set_index("trial").reindex(range(120)).p_match.values
            hb = boot[(cond, "unlimited")]
            vlm[cond] = {"cells": cell_accuracy((p > 0.5).astype(float)), "r_all": {k: vv for k, vv in boot_corr(p, hb).items() if k != "_rs"},
                         "r_same": {k: vv for k, vv in boot_corr(p, hb, np.where(SAME)[0]).items() if k != "_rs"}}

    # ---- best layer per model (by occluded same-category r) and the scale table
    summary_rows = []
    for name, m in models.items():
        best = max(m["layers"].items(), key=lambda kv: kv[1]["occluded"]["r_same"]["mean"] if np.isfinite(kv[1]["occluded"]["r_same"]["mean"]) else -9)
        L, e = best
        summary_rows.append({"model": name, "params": m["params"], "best_layer": L,
                             "occ_acc_same": e["occluded"]["cells"]["same_category"], "occ_acc_diff": e["occluded"]["cells"]["different_category"],
                             "unocc_acc_same": e["unoccluded"]["cells"]["same_category"],
                             "occ_r_same": e["occluded"]["r_same"]["mean"], "occ_r_same_ci": e["occluded"]["r_same"]["ci"],
                             "occ_r_all": e["occluded"]["r_all"]["mean"], "unocc_r_same": e["unoccluded"]["r_same"]["mean"],
                             "occ_margin_rt_spearman": e["occluded"]["margin_vs_log_rt"]["spearman"],
                             "dedrape_loo_acc_same": e["occluded"].get("dedrape_loo", {}).get("cells", {}).get("same_category"),
                             "dedrape_loco_acc_same": e["occluded"].get("dedrape_loco", {}).get("cells", {}).get("same_category"),
                             "dedrape_loco_r_same": e["occluded"].get("dedrape_loco", {}).get("r_same", {}).get("mean"),
                             "imagine_drape_acc_same": e["occluded"].get("imagine_drape", {}).get("cells", {}).get("same_category"),
                             "imagine_drape_r_same": e["occluded"].get("imagine_drape", {}).get("r_same", {}).get("mean"),
                             "imagine_undrape_r_same": e["occluded"].get("imagine_undrape", {}).get("r_same", {}).get("mean"),
                             "vs_pbas_p": e["occluded"]["vs_pbas_same"]["p_two_sided"]})
    summary = pd.DataFrame(summary_rows)

    res = {"human": {k: v for k, v in human.items() if k != "trial_table"}, "reference": {n: {c: {k: v for k, v in e.items() if not k.startswith("_")} for c, e in d.items()} for n, d in ref.items()},
           "models": models, "vlm": vlm, "summary": summary.to_dict("records")}
    figures(res, human, ref, models, out / "figures")
    with open(out / "cloth_results.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else (o.item() if hasattr(o, "item") else str(o)))
    md = summarize(res, summary)
    (out / "cloth_results.md").write_text(md)
    print(md)


def summarize(res, summary: pd.DataFrame) -> str:
    h = res["human"]
    f = lambda v, d=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    lines = ["# Shape under cloth: humans, PbAS and modern models", ""]
    lines.append(f"Humans: {h['n_workers_kept']} workers kept. Unlimited-time accuracy, occluded: all {f(h['cells']['occluded/unlimited']['all'])}, "
                 f"same-category {f(h['cells']['occluded/unlimited']['same_category'])}, different-category {f(h['cells']['occluded/unlimited']['different_category'])}; "
                 f"unoccluded same-category {f(h['cells']['unoccluded/unlimited']['same_category'])}.")
    lines.append(f"Split-half ceiling (unlimited): occluded all {f(h['ceiling']['occluded']['all']['mean'])}, occluded same-category "
                 f"{f(h['ceiling']['occluded']['same_category']['mean'])}, unoccluded same-category {f(h['ceiling']['unoccluded']['same_category']['mean'])}.")
    r = res["reference"]
    lines.append(f"PbAS (released runs, iteration 110): occluded same-category acc {f(r['PbAS']['occluded']['unlimited']['cells']['same_category'])}, "
                 f"trial r same-category {f(r['PbAS']['occluded']['unlimited']['r_same']['mean'])} [{f(r['PbAS']['occluded']['unlimited']['r_same']['ci'][0])}, {f(r['PbAS']['occluded']['unlimited']['r_same']['ci'][1])}], "
                 f"all trials {f(r['PbAS']['occluded']['unlimited']['r_all']['mean'])}; unoccluded same-category r {f(r['PbAS']['unoccluded']['unlimited']['r_same']['mean'])}; "
                 f"RT r (log) occluded {f(r['PbAS']['occluded']['rt_r_log'])}, unoccluded {f(r['PbAS']['unoccluded']['rt_r_log'])}. Paper: 0.44 / 0.44, RT 0.71 / 0.67.")
    lines += ["", "| model | params | best layer | occluded acc same / diff | unoccluded acc same | occluded r same [CI] | occluded r all | unocc r same | margin–RT ρ | de-drape acc same (loo / loco) | de-drape loco r same | imagine drape acc / r same | imagine undrape r same | p vs PbAS |",
              "|" + "---|" * 14]
    for _, s in summary.sort_values("occ_r_same", ascending=False).iterrows():
        lines.append(f"| {s.model} | {f(s.params/1e6,0) if s.params else '–'}M | {s.best_layer} | {f(s.occ_acc_same)} / {f(s.occ_acc_diff)} | {f(s.unocc_acc_same)} | "
                     f"{f(s.occ_r_same)} [{f(s.occ_r_same_ci[0])}, {f(s.occ_r_same_ci[1])}] | {f(s.occ_r_all)} | {f(s.unocc_r_same)} | {f(s.occ_margin_rt_spearman)} | "
                     f"{f(s.dedrape_loo_acc_same)} / {f(s.dedrape_loco_acc_same)} | {f(s.dedrape_loco_r_same)} | {f(s.imagine_drape_acc_same)} / {f(s.imagine_drape_r_same)} | {f(s.imagine_undrape_r_same)} | {f(s.vs_pbas_p,3)} |")
    if res["vlm"]:
        lines += ["", "VLM (Qwen3-VL 2AFC): " + json.dumps({c: {"acc_same": v["cells"]["same_category"], "acc_diff": v["cells"]["different_category"], "r_same": v["r_same"]["mean"]} for c, v in res["vlm"].items()})]
    lines += ["", "Readouts: BU = paper protocol on frozen embeddings; de-drape = linear map draped→bare learned on other trials (loo) or other categories (loco); "
              "imagine = generator-in-the-loop analysis-by-synthesis (if imagined images were produced). r values are means over participant bootstraps; CI 95%."]
    return "\n".join(lines)


def figures(res, human, ref, models, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(res["summary"])
    if len(summ) == 0:
        return
    summ = summ.sort_values("occ_r_same", ascending=False)
    # trial-level r on occluded same-category trials
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(summ) + 3), 4))
    x = np.arange(len(summ))
    ax.bar(x, summ.occ_r_same, yerr=[summ.occ_r_same - summ.occ_r_same_ci.apply(lambda c: c[0]), summ.occ_r_same_ci.apply(lambda c: c[1]) - summ.occ_r_same], color="#36c", label="BU readout")
    if summ.dedrape_loco_r_same.notna().any():
        ax.scatter(x, summ.dedrape_loco_r_same, color="#e80", zorder=3, label="de-drape (held-out category)")
    if summ.imagine_drape_r_same.notna().any():
        ax.scatter(x, summ.imagine_drape_r_same, color="#2a2", marker="s", zorder=3, label="imagination (drape)")
    ax.axhline(ref["PbAS"]["occluded"]["unlimited"]["r_same"]["mean"], ls="--", c="k", label="PbAS")
    ax.axhline(human["ceiling"]["occluded"]["same_category"]["mean"], ls=":", c="gray", label="split-half ceiling")
    ax.set_xticks(x); ax.set_xticklabels([f"{m}\n{l}" for m, l in zip(summ.model, summ.best_layer)], fontsize=7)
    ax.set_ylabel("r with human accuracy (occluded, same-category)"); ax.legend(fontsize=7); ax.set_ylim(-0.3, 0.9)
    fig.tight_layout(); fig.savefig(out / "trial_correlation_occluded_same.png", dpi=150); plt.close(fig)
    # cell accuracies
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(summ) + 3), 4))
    w = 0.25
    ax.bar(x - w, summ.occ_acc_same, w, label="occluded same-category", color="#c33")
    ax.bar(x, summ.occ_acc_diff, w, label="occluded different-category", color="#e99")
    ax.bar(x + w, summ.unocc_acc_same, w, label="unoccluded same-category", color="#9bd")
    hc = human["cells"]
    ax.axhline(hc["occluded/unlimited"]["same_category"], ls="--", c="#c33", lw=1, label="humans occluded same (unlimited)")
    ax.axhline(0.5, c="gray", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(summ.model, fontsize=8); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.set_ylim(0.3, 1.02)
    fig.tight_layout(); fig.savefig(out / "cell_accuracy.png", dpi=150); plt.close(fig)
    # layer sweeps
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, m in models.items():
        ls = list(m["layers"].items())
        ax.plot(range(len(ls)), [e["occluded"]["r_same"]["mean"] for _, e in ls], marker="o", label=name)
    ax.axhline(ref["PbAS"]["occluded"]["unlimited"]["r_same"]["mean"], ls="--", c="k")
    ax.set_xlabel("layer index (as stored)"); ax.set_ylabel("r (occluded, same-category)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "layer_sweep.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
