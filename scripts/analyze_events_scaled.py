"""Analysis of the scaled and cross-family signals, with the same pre-specified readouts as Pilot 0.

For every model signal in ``results/events_features_scaled.csv`` (plus the Pilot 0 signals from
``results/events_features.csv`` on the same bins, for a like-for-like comparison):

  B. boundary localization: within-scene AUROC for expert-boundary bins, mean over scenes, 95% scene
     bootstrap, and the paired difference against optical-flow acceleration;
  C. held-out human prediction: leave-one-scene-out logistic regression of trial-level detection on the
     low-level base (lab pixel change, pix_diff, flow_speed, flow_accel, t, t^2) with and without the
     signal; gain in held-out log-likelihood per trial (x1000) with scene-bootstrap CI; Holm over the
     new model signals (the Pilot 0 signals keep their original Holm family and are shown for reference).

Also writes a scale table (family, size, parameters, AUROC, gain) so the within-recipe scale curves
(H4, descriptive) can be read directly. Output: ``results/events_results_scaled.json`` and ``.md``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_events import LOW, boot_mean, boundary_auc, load_trials, loso_ll, scene_boot_diff, zwithin  # noqa: E402

from world_maps.core import holm_adjust  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

FAMILY = {"vjepa": "V-JEPA 2", "vmae": "VideoMAE", "dino": "DINOv2", "dinov3": "DINOv3", "siglip2": "SigLIP 2", "vlm": "Qwen3-VL"}
PILOT0 = {"vjepa_surprise": ("V-JEPA 2", "L", 300e6), "vmae_surprise": ("VideoMAE", "base", 87e6), "dino_change": ("DINOv2", "base", 86e6)}


def family_of(sig: str) -> str:
    stem = sig.split("_")[0]
    for k in sorted(FAMILY, key=len, reverse=True):
        if stem.startswith(k):
            return FAMILY[k]
    return stem


def main(scaled_csv: str, pilot_csv: str, meta_path: str) -> None:
    tr, _ = load_trials()
    sc = pd.read_csv(scaled_csv)
    p0 = pd.read_csv(pilot_csv)[["scene", "bin"] + LOW + list(PILOT0)]
    feat = p0.merge(sc, on=["scene", "bin"], how="inner").dropna().reset_index(drop=True)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    new_sigs = [c for c in sc.columns if c not in ("scene", "bin") and not c.endswith("_mismatch")]
    sigs = list(PILOT0) + new_sigs
    res = {"n_bins": int(len(feat)), "n_scenes": int(feat.scene.nunique()), "models": meta}

    # B
    ref = boundary_auc(feat, "flow_accel")
    resB = {}
    for s in sigs + ["flow_accel"]:
        per = boundary_auc(feat, s)
        m, lo, hi = boot_mean(per.values())
        common = [k for k in per if k in ref]
        resB[s] = dict(auc=m, lo=lo, hi=hi, minus_flow_accel=boot_mean([per[k] - ref[k] for k in common]))
    res["B_boundary_auc"] = resB
    # sanity for V-JEPA sizes: real future predicted better than another scene's future
    res["sanity_surprise_lt_mismatch"] = {s: float((feat[s] < feat[s.replace("_surprise", "_mismatch")]).mean())
                                          for s in new_sigs if s.endswith("_surprise") and s.replace("_surprise", "_mismatch") in feat}

    # C
    p = tr[tr.probe].merge(feat, on=["scene", "bin"], how="inner")
    p["t"] = p.bin * 0.2
    p["t2"] = p.t ** 2
    p["lab_pix"] = p.lab_pix.fillna(p.lab_pix.median())
    p = zwithin(p, sigs + LOW + ["lab_pix"])
    p[["t", "t2"]] = (p[["t", "t2"]] - p[["t", "t2"]].mean()) / p[["t", "t2"]].std()
    base = ["lab_pix", "pix_diff", "flow_speed", "flow_accel", "t", "t2"]
    ll0 = loso_ll(p, base)
    resC = {"n_trials": int(len(p)), "n_scenes": int(p.scene.nunique())}
    pv = {}
    for s in sigs:
        ll1 = loso_ll(p, base + [s])
        est, lo, hi, pval = scene_boot_diff(p, ll0, ll1)
        resC[s] = dict(dLL_per_trial_x1000=1000 * est, lo=1000 * lo, hi=1000 * hi, p=pval)
        if s in new_sigs:
            pv[s] = pval
    for s, adj in holm_adjust(pv).items():
        resC[s]["p_holm"] = adj
    res["C_heldout_human"] = resC

    # scale table
    rows = []
    for s in sigs:
        if s in PILOT0:
            fam, size, npar = PILOT0[s]
        else:
            name = s.split("_")[0]
            fam, size = family_of(s), name.replace(name.rstrip("Lg384H"), "") or "so400m"
            npar = meta.get(name, {}).get("params")
        rows.append({"signal": s, "family": fam, "size": size, "params_M": None if npar is None else round(npar / 1e6),
                     "boundary_auc": resB[s]["auc"], "auc_lo": resB[s]["lo"], "auc_hi": resB[s]["hi"],
                     "gain_x1000": resC[s]["dLL_per_trial_x1000"], "gain_lo": resC[s]["lo"], "gain_hi": resC[s]["hi"],
                     "p_holm": resC[s].get("p_holm")})
    table = pd.DataFrame(rows).sort_values(["family", "params_M"], na_position="last")
    res["scale_table"] = table.to_dict("records")
    json.dump(res, open(RESULTS / "events_results_scaled.json", "w"), indent=1)
    md = ["| Signal | Family | Size | Params (M) | Boundary AUROC [95% CI] | Held-out gain ×10⁻³ nats/trial [95% CI] | Holm p |", "|---|---|---|---|---|---|---|"]
    for r in table.to_dict("records"):
        ph = "ref" if r["p_holm"] is None else f"{r['p_holm']:.3f}"
        md.append(f"| {r['signal']} | {r['family']} | {r['size']} | {r['params_M'] or ''} | {r['boundary_auc']:.2f} [{r['auc_lo']:.2f}, {r['auc_hi']:.2f}] | "
                  f"{r['gain_x1000']:+.1f} [{r['gain_lo']:+.1f}, {r['gain_hi']:+.1f}] | {ph} |")
    md.append(f"\n{res['n_bins']} bins, {res['n_scenes']} scenes, {resC['n_trials']} probe trials. flow_accel boundary AUROC {resB['flow_accel']['auc']:.2f}. "
              f"Pilot 0 rows (L/base) are reference values on the same bins; Holm is over the new signals only.")
    (RESULTS / "events_results_scaled.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RESULTS / "events_features_scaled.csv"),
         sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "events_features.csv"),
         sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "events_models_scaled.json"))
