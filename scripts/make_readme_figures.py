"""Figures for the README, built from the committed result files.

Outputs docs/figures/{cloth_gap,physion_error_agreement,events_scale,plane}.png and two cropped contact sheets.
Run: python scripts/make_readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREEN, RED, GREY, ORANGE = "#3b6fd1", "#2f9e44", "#c92a2a", "#868e96", "#e8590c"
NAMES = {"vjepaL": "V-JEPA 2-L", "vjepaH": "V-JEPA 2-H", "dinoB": "DINOv2-B", "dinoL": "DINOv2-L", "dinog": "DINOv2-g",
         "siglip2": "SigLIP 2", "clipL": "CLIP-L", "alexnet": "AlexNet", "vmaeL": "VideoMAE-L"}


def cloth():
    c = json.load(open(RES / "cloth" / "cloth_results.json"))
    rows = sorted(c["summary"], key=lambda r: -r["occ_r_same"])
    hum = c["human"]["cells"]["occluded/unlimited"]["same_category"]
    ceil = c["human"]["ceiling"]["occluded"]["same_category"]["mean"]
    pbas = c["reference"]["PbAS"]["occluded"]["unlimited"]
    labels = [NAMES.get(r["model"], r["model"]) for r in rows]
    x = np.arange(len(rows))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.bar(x, [r["occ_acc_same"] for r in rows], color=BLUE, label="frozen encoder, paper protocol")
    a.scatter(x, [r["dedrape_loco_acc_same"] for r in rows], color=ORANGE, zorder=3, label="linear de-draping map (held-out category)")
    a.scatter(x, [r["imagine_drape_acc_same"] for r in rows], color=GREEN, marker="s", zorder=3, label="image generator in the loop (SDXL)")
    a.axhline(hum, color=RED, ls="--", lw=1.2, label=f"humans {hum:.2f}")
    a.axhline(pbas["cells"]["same_category"], color="k", ls=":", lw=1.2, label=f"PbAS {pbas['cells']['same_category']:.2f}")
    a.axhline(0.5, color=GREY, lw=0.8)
    a.set_ylim(0.25, 1.0); a.set_ylabel("accuracy"); a.set_title("Hard trials (object under cloth, same-category distractor)")
    a.set_xticks(x); a.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); a.legend(fontsize=7, loc="upper right")
    r = np.array([r["occ_r_same"] for r in rows]); lo = np.array([r_["occ_r_same_ci"][0] for r_ in rows]); hi = np.array([r_["occ_r_same_ci"][1] for r_ in rows])
    b.bar(x, r, yerr=[r - lo, hi - r], color=BLUE, capsize=3, label="frozen encoder")
    b.scatter(x, [r_["dedrape_loco_r_same"] for r_ in rows], color=ORANGE, zorder=3, label="de-draping map")
    b.scatter(x, [r_["imagine_drape_r_same"] for r_ in rows], color=GREEN, marker="s", zorder=3, label="image generator")
    b.axhline(pbas["r_same"]["mean"], color="k", ls=":", lw=1.2, label=f"PbAS {pbas['r_same']['mean']:.2f}")
    b.axhline(ceil, color=GREY, ls="--", lw=1.2, label=f"split-half ceiling {ceil:.2f}")
    b.axhline(0, color=GREY, lw=0.8)
    b.set_ylim(-0.3, 0.85); b.set_ylabel("trial-level r with human accuracy"); b.set_title("Do the models find the same trials hard as people?")
    b.set_xticks(x); b.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); b.legend(fontsize=7, loc="upper right")
    fig.suptitle("Shape under cloth: no modern encoder closes the gap to PbAS", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "cloth_gap.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def single_participant_ceiling(n_draws: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Partial r (given the label) of one idealised participant: one Bernoulli(P(yes)) draw per stimulus."""
    st = pd.read_csv(RES / "physion" / "stimuli.csv")
    h = st["p_yes_human"].to_numpy(float); y = st["label"].astype(str).str.lower().eq("true").to_numpy(float)
    rng = np.random.default_rng(seed)
    def partial(m):
        rm = m - np.polyval(np.polyfit(y, m, 1), y); rh = h - np.polyval(np.polyfit(y, h, 1), y)
        return float(np.corrcoef(rm, rh)[0, 1])
    vals = np.array([partial((rng.random(len(h)) < h).astype(float)) for _ in range(n_draws)])
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def physion():
    p = json.load(open(RES / "physion" / "physion_results.json"))
    ceil, ceil_lo, ceil_hi = single_participant_ceiling()
    pick = [("vjepaH/last_grid/within", "V-JEPA 2-H"), ("vjepaH/last_grid/loso", "V-JEPA 2-H\n(held-out scenario)"),
            ("vjepaL/last_slot/within", "V-JEPA 2-L"), ("vmaeL/last_grid/within", "VideoMAE-L"), ("siglip2/last/within", "SigLIP 2"),
            ("dinoL/last_minus_first/within", "DINOv2-L\n(motion diff.)"), ("vlm", "Qwen3-VL\nyes/no"), ("kinematic/within", "kinematic\nheuristic")]
    vals, los, his, accs, labels, cols = [], [], [], [], [], []
    for k, lab in pick:
        a = p["readouts"][k]["all"]
        vals.append(a["partial_r_given_label"]); los.append(a["partial_r_given_label_ci"][0]); his.append(a["partial_r_given_label_ci"][1])
        accs.append(a["accuracy"]); labels.append(lab); cols.append(BLUE)
    g = p["generator"]["ltx"]["p_cal"]
    for key, lab in [("all", "LTX generator\n(imagined outcome)"), ("primary", "LTX generator\n(validated scenarios)")]:
        a = g[key]
        vals.append(a["partial_r_given_label"]); los.append(a["partial_r_given_label_ci"][0]); his.append(a["partial_r_given_label_ci"][1])
        accs.append(a["accuracy"]); labels.append(lab); cols.append(GREEN)
    vals, los, his = np.array(vals), np.array(los), np.array(his)
    x = np.arange(len(vals))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    a.bar(x, vals, yerr=[vals - los, his - vals], color=cols, capsize=3)
    a.axhline(0, color="k", lw=1, label="accuracy-matched random agent (0.00)")
    a.axhline(ceil, color=RED, ls="--", lw=1.2, label=f"one idealised participant ({ceil:.2f} [{ceil_lo:.2f}, {ceil_hi:.2f}])")
    a.set_ylim(-0.1, 0.7)
    a.set_xticks(x); a.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=7.5, rotation=30, ha="right")
    a.set_ylabel("partial r with human P(yes), label removed")
    a.set_title("Who gets the same Physion clips wrong as people?"); a.legend(fontsize=8, loc="upper right")
    # accuracy vs human-likeness for every readout
    pts = [(v["all"]["accuracy"], v["all"]["partial_r_given_label"], k) for k, v in p["readouts"].items()]
    b.scatter([q[0] for q in pts], [q[1] for q in pts], color=BLUE, s=28, label="encoder / VLM / heuristic readouts")
    b.scatter([g["all"]["accuracy"]], [g["all"]["partial_r_given_label"]], color=GREEN, s=80, marker="s", label="LTX generator (all scenarios)")
    b.scatter([g["primary"]["accuracy"]], [g["primary"]["partial_r_given_label"]], color=GREEN, s=80, marker="^", label="LTX generator (validated scenarios)")
    b.axhline(0, color="k", lw=1); b.axvline(0.5, color=GREY, lw=0.8)
    hum = np.mean(list(p["human"]["accuracy_by_scenario"].values()))
    b.axvline(hum, color=RED, ls="--", lw=1.2, label=f"human accuracy {hum:.2f}")
    b.set_xlabel("accuracy vs ground truth"); b.set_ylabel("agreement with human errors (partial r)")
    b.set_title("Agreement with human errors vs accuracy, all readouts"); b.legend(fontsize=7, loc="upper left")
    fig.suptitle("Physion: a probe on frozen V-JEPA 2 features predicts which clips people get wrong; imagined outcomes carry a weaker signal", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "physion_error_agreement.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def events():
    e = json.load(open(RES / "events_results_scaled.json"))
    t = e["scale_table"]
    fam_cols = {"DINOv2": BLUE, "SigLIP 2": "#1c7ed6", "V-JEPA 2": RED, "VideoMAE": ORANGE, "Qwen3-VL": GREY}
    order = ["dino_change", "dinoL_change", "dinog_change", "siglip2_change", "vjepa_surprise", "vjepaH_surprise", "vjepag_surprise",
             "vmae_surprise", "vmaeL_surprise", "vlm_boundary"]
    rows = [r for s in order for r in t if r["signal"] == s]
    labels = [f"{r['family']}\n{'8B' if r['family'] == 'Qwen3-VL' else r['size']}" for r in rows]
    x = np.arange(len(rows))
    g = np.array([r["gain_x1000"] for r in rows]); lo = np.array([r["gain_lo"] for r in rows]); hi = np.array([r["gain_hi"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x, g, yerr=[g - lo, hi - g], color=[fam_cols[r["family"]] for r in rows], capsize=3)
    ax.set_ylim(-4, 36)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("held-out gain in predicting human slowdown misses\n(×10⁻³ nats per trial, beyond motion covariates)")
    ax.set_title("Physical events at scale: static frame change predicts human misses at every size;\nvideo-model prediction error does not (* Holm p < .05)", fontsize=11)
    for i, r in enumerate(rows):
        if r.get("p_holm") == r.get("p_holm") and r.get("p_holm") is not None and r["p_holm"] < 0.05:
            ax.text(i, hi[i] + 0.8, "*", ha="center", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "events_scale.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def plane():
    """Accuracy x human-likeness plane for both studies (best layer / pooling per model, as in the tables)."""
    c = json.load(open(RES / "cloth" / "cloth_results.json"))
    p = json.load(open(RES / "physion" / "physion_results.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    off = {"dinog": (4, -9), "siglip2": (-44, 6), "vjepaH": (4, -9)}
    fixed_layer = {"alexnet": "fc7", "vjepaL": "L24_grid", "vjepaH": "L32_grid"}
    for s_ in c["summary"]:
        m = s_["model"]; L = c["models"][m]["layers"][s_["best_layer"]]["occluded"]
        x, y = L["cells"]["same_category"], L["r_same"]["mean"]
        a.scatter(x, y, c=BLUE, s=46, zorder=3); a.annotate(NAMES[m], (x, y), fontsize=7, xytext=off.get(m, (4, 3)), textcoords="offset points")
        F = c["models"][m]["layers"][fixed_layer.get(m, "final_pooled")]["occluded"]
        a.scatter(F["cells"]["same_category"], F["r_same"]["mean"], marker="x", c="#555555", s=30, zorder=3, linewidths=1)
        dd = L["dedrape_loco"]; a.scatter(dd["cells"]["same_category"], dd["r_same"]["mean"], marker="o", facecolors="none", edgecolors=ORANGE, s=46, zorder=3)
        im = L["imagine_drape"]; a.scatter(im["cells"]["same_category"], im["r_same"]["mean"], marker="s", c=GREEN, s=30, zorder=3)
    pb = c["reference"]["PbAS"]["occluded"]["unlimited"]; hum = c["human"]["cells"]["occluded/unlimited"]["same_category"]
    ceil = c["human"]["ceiling"]["occluded"]["same_category"]["mean"]
    a.scatter(pb["cells"]["same_category"], pb["r_same"]["mean"], marker="*", c="k", s=150, zorder=4)
    a.annotate("PbAS (physics engine)", (pb["cells"]["same_category"], pb["r_same"]["mean"]), fontsize=8, xytext=(-108, 6), textcoords="offset points")
    a.axhline(0, color=GREY, lw=1, ls="--"); a.axhline(ceil, color=GREY, lw=1, ls=":"); a.text(0.60, ceil - 0.035, f"split-half ceiling {ceil:.2f}", fontsize=7.5, color=GREY)
    a.axvline(hum, color=GREY, lw=1, ls=":"); a.text(hum + 0.007, 0.55, f"humans {hum:.2f}", fontsize=7.5, color=GREY, rotation=90)
    a.axvline(0.5, color="lightgray", lw=1); a.text(0.503, 0.28, "chance", fontsize=7, color=GREY)
    a.text(0.31, -0.05, "accuracy-matched random agent: r = 0", fontsize=7.5, color=GREY)
    a.set_xlim(0.3, 0.85); a.set_ylim(-0.25, 0.8); a.set_xlabel("accuracy, occluded same-category trials"); a.set_ylabel("trial-level r with human accuracy")
    a.set_title("Shape under cloth (60 hard trials, 25 people per cell)", fontsize=11)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="frozen encoder, paper protocol (best layer, selected on r)"),
                      Line2D([], [], marker="x", color="#555555", ls="", label="same encoder at the final layer (fixed convention)"),
                      Line2D([], [], marker="o", markerfacecolor="none", color=ORANGE, ls="", label="linear de-draping map (held-out category)"),
                      Line2D([], [], marker="s", color=GREEN, ls="", label="SDXL in the analysis-by-synthesis loop"),
                      Line2D([], [], marker="*", color="k", ls="", markersize=10, label="PbAS")], fontsize=7.5, loc="upper left")
    lab = {"vjepaH/last_grid/within": ("V-JEPA 2-H probe", (4, 3)), "vjepaH/last_grid/loso": ("V-JEPA 2-H probe (held-out scenario)", (-150, -4)),
           "vjepaL/last_slot/within": ("V-JEPA 2-L probe", (4, 3)), "vmaeL/last_grid/within": ("VideoMAE-L probe", (4, 3)),
           "vmaeL/last_grid/loso": ("VideoMAE-L probe (held-out)", (4, 2)), "siglip2/last/within": ("SigLIP 2 probe", (4, 3)),
           "dinoL/last_minus_first/within": ("DINOv2-L probe", (4, 4)), "vlm": ("Qwen3-VL", (-44, -10)), "kinematic/within": ("kinematic heuristic", (4, 3))}
    for k, (l, o) in lab.items():
        r_ = p["readouts"][k]["all"]; b.scatter(r_["accuracy"], r_["partial_r_given_label"], c=BLUE, s=46, zorder=3)
        b.annotate(l, (r_["accuracy"], r_["partial_r_given_label"]), fontsize=7, xytext=o, textcoords="offset points")
    g = p["generator"]["ltx"]["p_cal"]
    b.scatter(g["all"]["accuracy"], g["all"]["partial_r_given_label"], marker="s", c=GREEN, s=46, zorder=3)
    b.annotate("LTX imagined outcome (all scenarios)", (g["all"]["accuracy"], g["all"]["partial_r_given_label"]), fontsize=7, xytext=(4, -9), textcoords="offset points")
    b.scatter(g["primary"]["accuracy"], g["primary"]["partial_r_given_label"], marker="^", c=GREEN, s=60, zorder=3)
    b.annotate("LTX imagined outcome\n(validated scenarios)", (g["primary"]["accuracy"], g["primary"]["partial_r_given_label"]), fontsize=7, xytext=(8, -30), textcoords="offset points", ha="left")
    sp, _, _ = single_participant_ceiling()
    b.axhline(sp, color=RED, lw=1, ls="--"); b.text(0.455, sp + 0.01, f"one idealised participant {sp:.2f} (reference, not a ceiling for graded outputs)", fontsize=7.5, color=RED)
    b.axhline(0, color=GREY, lw=1, ls="--"); b.text(0.455, 0.012, "accuracy-matched random agent: partial r = 0", fontsize=7.5, color=GREY)
    hum_p = np.mean(list(p["human"]["accuracy_by_scenario"].values()))
    b.axvline(hum_p, color=GREY, lw=1, ls=":"); b.text(hum_p + 0.005, 0.25, f"humans {hum_p:.2f}", fontsize=7.5, color=GREY, rotation=90)
    b.axvline(0.5, color="lightgray", lw=1); b.text(0.503, 0.45, "chance", fontsize=7, color=GREY)
    b.set_xlim(0.45, 0.8); b.set_ylim(-0.08, 0.66); b.set_xlabel("accuracy vs ground truth (1.5 s context)"); b.set_ylabel("partial r with human P(yes), label removed")
    b.set_title("Physion (1,144 clips, 66-96 people each)", fontsize=11)
    b.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="probe on frozen features / VLM / heuristic"),
                      Line2D([], [], marker="s", color=GREEN, ls="", label="LTX-Video imagined outcome, all"),
                      Line2D([], [], marker="^", color=GREEN, ls="", label="LTX-Video imagined outcome, validated")], fontsize=7.5, loc="lower right")
    for ax in (a, b):
        ax.grid(alpha=0.25)
    fig.suptitle("The accuracy x human-likeness plane: accurate is not the same as human-like", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "plane.png", dpi=170); plt.close(fig)


def sheets():
    im = Image.open(RES / "cloth" / "imagined_contact_sheet.png")
    im.save(OUT / "cloth_imagined.png")
    sheet = Image.open(RES / "physion" / "generated" / "ltx" / "last_frames_sheet.png")
    w = sheet.width // 12
    sheet.crop((0, 0, 6 * w, sheet.height)).save(OUT / "physion_generated_last_frames.png")


if __name__ == "__main__":
    cloth()
    physion()
    events()
    plane()
    sheets()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
