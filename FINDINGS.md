# Findings

Status: exploratory analyses of public data, October 23, 2026. Corrections are logged at the end.

## 1. Physical events (Pilot 0)

### Data

[Yates, Yasuda & Yildirim (2024)](https://cncl.yale.edu/assets/pdf/2024_YatesetalEvents.pdf), OSF project [ma32p](https://osf.io/ma32p/), Experiment 1.

- **Videos:** 32 base videos. Each combines two of collision, falling, containment, and occlusion, and 14 videos contain each type.
- **Expert boundaries:** 18 collision, 18 falling, 17 containment, and 27 occlusion boundary bins.
- **Task:** participants pressed a key on detecting a brief slowdown (158–190 ms at half frame rate), placed at one 200 ms position per trial.

### Reproduction

The rules of the lab's analysis notebook reproduce the published effect exactly:

| | Value |
|---|---|
| Participants retained / in the paired contrast | 109 / 108 |
| Detection at boundaries vs elsewhere | 33.2% vs 61.1% |
| Paired test | t(107) = −11.36, d_z = −1.09 (sample SD; −1.098 with the notebook's population SD) |
| Catch trials with a keypress | about 10% before the lab's response-time filter, 1.2% after |

### Signals

All signals were computed from the original, slowdown-free videos at every 200 ms position: 1,515 extracted, 1,509 analyzed after dropping final bins whose window passes the clip end.

| Signal | Definition |
|---|---|
| `vjepa_surprise` (primary) | V-JEPA 2-L (`facebook/vjepa2-vitl-fpc64-256`). Context encoder on the 12 frames (15 fps) before *t*; predictor forecasts the 4 frames after *t*; mean L1 against layer-normed encoder tokens of the real 16-frame clip |
| `vjepa_surprise_top5` | mean of the top 5% per-token errors |
| `vjepa_change` | 1 − cosine of mean encoder tokens, tubelets 4–5 vs 6–7 |
| `vmae_surprise` | VideoMAE-base (`MCG-NJU/videomae-base`) reconstruction loss with the two future tubelets masked |
| `dino_change` | DINOv2-base CLS, 1 − cosine between frames at *t* − 0.2 s and *t* + 0.2 s |
| `vjepa_mismatch` (sanity check) | V-JEPA 2 forecast scored against a different scene's clip |
| Low-level covariates | mean absolute frame difference, Farneback flow speed and acceleration, time, time², and the lab's pixel-change measure (computed from the probe videos) |

Frames are letterboxed to a square by edge replication. One scene (`falling_falling4316`) ships only as countdown-prefixed 30 fps files; its countdown is stripped automatically and frames are repeated to the 120 fps grid.

### Prespecified analysis

- **Boundary marking:** within-scene AUROC for expert-boundary bins, averaged over scenes.
- **Human prediction:** leave-one-scene-out logistic regression of trial-level detection on the low-level covariates, with and without each signal. Signals are z-scored within scene.
- **Inference:** 95% scene-bootstrap intervals; Holm correction across the five model signals.
- **Primary:** `vjepa_surprise`.

| Signal | Boundary AUROC | Δ held-out log-likelihood (×10⁻³ nats/trial) | Holm p |
|---|---|---|---|
| V-JEPA 2 prediction error (primary) | 0.65 [0.59, 0.71] | +0.5 [−1.5, +2.4] | 1.0 |
| V-JEPA 2 localized error | 0.53 [0.47, 0.60] | −0.5 [−1.2, +0.1] | 0.64 |
| V-JEPA 2 encoder change | 0.51 [0.44, 0.58] | +0.1 [−1.9, +2.1] | 1.0 |
| VideoMAE reconstruction error | 0.41 [0.36, 0.46] | −0.2 [−1.5, +0.9] | 1.0 |
| DINOv2 frame change | 0.79 [0.72, 0.85] | +17.4 [+9.1, +26.4] | 0.0025* |
| Flow acceleration | 0.69 [0.63, 0.75] | in baseline | — |
| Expert boundary label | — | +3.0 [−1.0, +6.8] | — |
| V-JEPA 2 vs a different scene's clip | 0.52 [0.45, 0.59] | — | — |

\* The bootstrap's resolution limit (0.0005) times 5.

**Other checks:**
- V-JEPA 2's forecast is closer to the true future than to a different scene's clip in 86% of bins.
- Adding DINOv2 change cuts the adjusted boundary log-odds coefficient by 59% [29, 153]. This is descriptive coefficient attenuation, not explained variance or mediation. V-JEPA 2 cuts it by 4% [−5, 18].
- Both coefficients are negative, as predicted: a larger signal means lower detection (−0.48 for DINOv2, −0.12 for V-JEPA 2).

### Sensitivity (added after review)

Within-scene z-scoring uses each held-out video's own covariate distribution. That step is transductive on covariates, though no labels are involved, so we reran the analysis with training-fold-only scaling and imputation.

| Analysis | DINOv2 gain | V-JEPA 2 gain | DINOv2 − V-JEPA 2 |
|---|---|---|---|
| Original (within-scene z) | +17.4 [9.1, 26.0] | +0.5 [−1.4, +2.4] | — |
| Training-fold scaling, scene bootstrap | +8.9 [1.8, 15.4] | −1.1 [−2.0, −0.4] | +10.0 [3.0, 16.9] |
| Same, crossed participant × scene bootstrap | +8.9 [−0.9, 16.8] | −1.1 [−2.2, −0.4] | +10.0 [0.5, 19.1] |
| Leave-one-event-pair-family-out (16 families) | +6.8 [−6.8, 17.1] | −1.6 [−3.5, −0.4] | +8.4 [−4.0, 19.5] |
| Without the 30 fps fallback video | +8.1 [1.6, 14.2] | −1.1 [−2.3, −0.3] | — |
| Only bins with a full 0.8 s context | +6.9 [1.4, 12.6] | −1.1 [−2.3, −0.3] | — |

All bootstraps reweight fixed held-out scores without refitting.

### Time alignment

We matched each probe video frame by frame against its original video: 61 probes, two in each of 29 scenes and one in each of 3. In 58, the slowdown begins within 0.1 s of 0.2*k* s; the median lag is +0.07 s, partly from the detection rule. The three outliers (+1.0, −0.3, −1.7 s) each sit in a scene whose other probe aligned.

### Exploratory

**Boundary AUROC by event type** (descriptive; no subgroup intervals):

| | Collision | Falling | Containment | Occlusion |
|---|---|---|---|---|
| DINOv2 change | 0.55 | 0.78 | 0.90 | 0.90 |
| Flow acceleration | 0.74 | 0.63 | 0.66 | 0.67 |
| V-JEPA 2 prediction error | 0.58 | 0.74 | 0.64 | 0.67 |

Occlusion boundaries include both disappearance and reappearance.

**Lag profile:** within-scene partial rank correlation of the detection rate at bin *k* with the signal at bin *k + L*.
- Flow acceleration one bin *after* the probe (L = +1) predicts lower detection: r = −0.22 [−0.32, −0.13]. This is consistent with the published dip about 200 ms before boundaries.
- V-JEPA 2 error at L = 0 gives r = −0.10 [−0.17, −0.02].

### Reading

V-JEPA 2's prediction error tracks event structure but did not explain human detection beyond low-level motion in this configuration. A static image model's frame change did better, robustly against V-JEPA 2 under scene-level and crossed resampling, but not when whole event-pair families are held out.

The mechanism is open. Changes in what is visible are one candidate explanation, and anticipatory updating is another. The planned confirmatory study crosses dynamic surprise with visible-state change on new stimuli to separate them.

## 2. Woven reanalysis

[Bi et al. (2025)](https://www.nature.com/articles/s41467-025-61458-x), OSF project [29jnd](https://osf.io/29jnd/) (CC0): 100 participants × 56 stiffness trials and 100 × 54 mass trials; 80 movies; 10 released model iterations per trial.

| | Stiffness | Mass |
|---|---|---|
| Human accuracy | 66.0% | 50.7% |
| Woven agreement with human choices | 64.4% | 62.8% |
| Task-trained DNN agreement | 52.4% | 48.6% |
| Woven − DNN (raw) | **+12.0 points** | **+14.1 points** |
| Woven − DNN (add-half smoothed iteration frequencies) | +10.9 | +12.9 |
| Resolvable difference, participants × all three movies (2.8 × bootstrap SE) | 8.4 | 7.5 |
| Same at a Holm α of 0.05/7 | 10.6 | 9.5 |

The resolvable difference is a normal-approximation sensitivity estimate, not a power calculation.

## 3. Shape under cloth

[Yildirim et al. (2024)](https://www.nature.com/articles/s41562-023-01759-7). Stimuli: [CNCLgithub/intuitive-physics-3d-shape-perception-stimuli](https://github.com/CNCLgithub/intuitive-physics-3d-shape-perception-stimuli). Behaviour and PbAS runs: [CNCLgithub/PbAS-model-human-comparisons](https://github.com/CNCLgithub/PbAS-model-human-comparisons) (20,880 trials; 162 of 174 workers kept by the lab's rule, accuracy > 71/120). Full table: [results/cloth/cloth_results.md](results/cloth/cloth_results.md); figures in `results/cloth/figures`; imagined images in `results/cloth/imagined_contact_sheet.png`.

### Reproduction of the paper's numbers

| Quantity | Ours | Paper |
|---|---|---|
| Occluded, unlimited: accuracy same / different category | 0.73 / 0.97 | Fig. 4a |
| Split-half ceiling, occluded, all / same-category | 0.80 / 0.71 | 0.80 / 0.71 |
| PbAS trial-level r, occluded same-category (iteration 110) | 0.44 [0.33, 0.54] | 0.44 [0.33, 0.54] |
| PbAS iterations-to-criterion vs log RT, occluded / unoccluded | 0.74 / 0.81 | 0.71 / 0.67 |
| AlexNet (bottom-up), occluded same-category accuracy / r | 0.53 / 0.08 | chance / 0.11 |

### Readouts

- **BU (paper protocol):** Pearson correlation between the target's embedding and each candidate's; choice = larger; Luce score for trial-level r; margin = |d1 − d2|.
- **De-draping:** ridge map from the draped target's embedding (PCA 100) to bare-object embeddings (PCA 64 on the 240 candidates), learned on the other trials (loo) or the other nine categories (loco); same 2AFC on the mapped embedding.
- **Imagination:** SDXL img2img drapes each candidate (K = 4) or un-drapes the target; candidates scored by feature similarity to the imagined images; sequential sampling over imagined samples gives an RT prediction.
- **VLM:** Qwen3-VL-8B 2AFC, both candidate orders.

### Results (best layer per model by occluded same-category r; participant bootstrap, 2,000)

| model | params | occluded acc same / diff | unoccluded acc same | occluded r same [CI] | margin–RT ρ | de-drape acc same (loo / loco) | de-drape loco r | imagine drape acc / r | p vs PbAS |
|---|---|---|---|---|---|---|---|---|---|
| vjepaL (L16) | 326M | 0.60 / 0.62 | 0.67 | 0.18 [0.11, 0.25] | −0.12 | 0.48 / 0.58 | −0.01 | 0.47 / −0.03 | < .001 |
| dinoB (L9) | 87M | 0.35 / 0.53 | 0.48 | 0.16 [0.07, 0.25] | −0.03 | 0.67 / 0.67 | −0.12 | 0.38 / 0.00 | .001 |
| siglip2 | 1136M | 0.53 / 0.58 | 0.75 | 0.10 [0.01, 0.19] | −0.35 | 0.57 / 0.60 | −0.09 | 0.42 / −0.16 | < .001 |
| alexnet (fc7) | 61M | 0.53 / 0.55 | 0.63 | 0.08 [0.01, 0.17] | −0.34 | 0.67 / 0.57 | 0.14 | 0.52 / 0.05 | < .001 |
| dinog | 1136M | 0.53 / 0.68 | 0.67 | 0.06 [−0.04, 0.15] | −0.32 | 0.53 / 0.62 | 0.08 | 0.45 / −0.01 | < .001 |
| clipL (L20) | 428M | 0.65 / 0.58 | 0.72 | 0.06 [−0.03, 0.14] | −0.08 | 0.62 / 0.55 | 0.05 | 0.55 / 0.02 | < .001 |
| dinoL (L20) | 304M | 0.40 / 0.65 | 0.63 | 0.04 [−0.04, 0.12] | −0.14 | 0.63 / 0.48 | −0.02 | 0.45 / −0.09 | < .001 |
| vjepaH (L24) | 654M | 0.62 / 0.62 | 0.68 | 0.03 [−0.04, 0.11] | −0.41 | 0.60 / 0.65 | 0.05 | 0.50 / 0.09 | < .001 |
| Qwen3-VL 2AFC | 8B | 0.53 / 0.38 | 0.75 | 0.25 | — | — | — | — | — |

Decision rules (set before the run): gap closed if any lower bound ≥ 0.33 → **not met**. Decodable-but-not-used if held-out-category accuracy ≥ 0.60 with r upper bound < 0.33 → **met** (dinoB, dinog, vjepaH, clipL). Imagination helps if ≥ 0.10 accuracy gain on the hard cell → **not met** (SDXL wraps texture around the object instead of draping; un-draping fails). Margin–RT: four of eight models at ρ ≤ −0.32; intervals not yet computed. DINOv3 was not run (gated checkpoint). The video-generation route (LTX-Video) produced no clips in its first run.

### Limits

Frozen embeddings and the paper's correlation readout; no fine-tuning. The generator keeps the candidate's pose, PbAS searched rotations. Participant bootstraps condition on the released trials.

## 4. Physical events at scale

Same protocol and bins as Pilot 0 (§1) on larger models and other families: 1,509 bins, 32 scenes, 6,239 probe trials. [results/events_results_scaled.md](results/events_results_scaled.md).

| Signal | Size | Boundary AUROC | Held-out gain ×10⁻³ nats/trial | Holm p |
|---|---|---|---|---|
| DINOv2 frame change | base (Pilot 0) | 0.79 [0.72, 0.85] | +17.4 [+8.9, +25.8] | — |
| DINOv2 frame change | L | 0.82 [0.76, 0.87] | **+20.9 [+10.5, +30.8]** | .005 |
| DINOv2 frame change | g | 0.82 [0.76, 0.87] | **+19.3 [+9.2, +29.2]** | .005 |
| SigLIP 2 frame change | so400m | 0.80 [0.74, 0.85] | +10.8 [+3.3, +18.7] | .016 |
| Qwen3-VL "new event begins?" | 8B | 0.74 [0.69, 0.78] | +2.8 [−1.1, +6.6] | 1.0 |
| V-JEPA 2 prediction error | L / H / g | 0.65 / 0.63 / 0.62 | +0.5 / +1.7 / +0.4 (all n.s.) | 1.0 |
| VideoMAE reconstruction error | base / L | 0.41 / 0.42 | −0.2 / +0.1 | 1.0 |

The dissociation of Pilot 0 holds across model sizes: static feature change predicts human misses; the predictive video models' surprise does not, at any scale tested.

## 5. Physion: generators as simulators

[Physion](https://github.com/cogtoolslab/physics-benchmarking-neurips2021) (Bear et al. 2021; MIT). 1,144 test clips with human P(yes) from 66–96 participants (reliability 0.99). People saw the first 1.5 s. Full table with paired comparisons: [results/physion/physion_results.md](results/physion/physion_results.md); clips and detector scores in `results/physion/generated/ltx`.

### Measurement

Per-stimulus colour models (red agent, yellow patient) calibrated on the released first-frame maps; contact relative to the end of the context (approach within 2 px, overlap gain > 10 %, or vanishing inside a container). On the **real** outcome frames: 0.81 accuracy overall (dominoes 0.88, support 0.90, roll 0.90, collide 0.78, drop 0.73, drape 0.74, contain 0.66, link 0.58). Scenarios ≥ 0.75 form the primary set. A per-scenario logistic calibration of detector features, fitted on real outcomes only, converts detector outputs to probabilities (CV accuracy 0.95 / 0.93 / 0.89 / 0.85 / 0.77 / 0.72 / 0.68 / 0.64).

### Readouts and metrics

Encoders (V-JEPA 2 L/H, VideoMAE-L, DINOv2-L, SigLIP 2) with a logistic readout on PCA-128 features, within-scenario 5-fold CV and leave-one-scenario-out; Qwen3-VL yes/no; kinematic heuristic. Generator: LTX-Video 2B continues the context (97 frames at 24 fps, 33 conditioning frames), 40 stimuli per scenario balanced on the label, one sample. Metrics: accuracy; r with human P(yes); κ with the human majority; partial r after regressing the label out of both (agreement on which stimuli are judged wrongly); accuracy-matched random agents as null; stimulus bootstraps (1,000).

### Results

| readout (1.5 s context) | n | accuracy | r with human P(yes) | κ | partial r given label [CI] (null 0.00) |
|---|---|---|---|---|---|
| V-JEPA 2-H last-slot grid, within-scenario | 1144 | 0.64 | 0.48 | 0.37 | **0.37 [0.32, 0.42]** |
| V-JEPA 2-H, leave-one-scenario-out | 1144 | 0.57 | 0.19 | 0.20 | 0.14 [0.08, 0.20] |
| V-JEPA 2-L last slot, within | 1144 | 0.63 | 0.43 | 0.34 | 0.29 [0.24, 0.35] |
| VideoMAE-L, within | 1144 | 0.60 | 0.34 | 0.23 | 0.26 [0.21, 0.32] |
| SigLIP 2 last frame, within | 1144 | 0.58 | 0.31 | 0.25 | 0.23 [0.17, 0.28] |
| DINOv2-L motion difference, within | 1144 | 0.61 | 0.27 | 0.24 | 0.18 [0.13, 0.24] |
| Qwen3-VL yes/no | 1144 | 0.59 | 0.22 | 0.17 | 0.12 [0.06, 0.17] |
| kinematic heuristic | 1144 | 0.52 | 0.04 | 0.02 | 0.03 [−0.02, 0.09] |
| **LTX-Video imagined outcome (calibrated)**, all scenarios | 320 | 0.58 | 0.17 | 0.13 | 0.07 [−0.03, 0.17] |
| **LTX-Video imagined outcome (calibrated)**, validated scenarios | 200 | — | — | — | **0.16 [0.04, 0.29]** |

Paired on the generator's subset, validated scenarios (generator minus readout, partial r): vs kinematic +0.20 [+0.07, +0.35]; vs DINOv2-L (loso) +0.24 [+0.07, +0.40]; vs SigLIP 2 (loso) +0.21 [+0.03, +0.39]; vs V-JEPA 2-H (within) −0.13 [−0.29, +0.05]; on all scenarios vs V-JEPA 2-H −0.35 [−0.48, −0.22]. The generator's accuracy profile across scenarios correlates with the human profile at Spearman 0.68; its error agreement is highest in collide (0.30) and support (0.32).

Decision rules (set before the run): H1 generator > best encoder on validated scenarios → **not met**; H2 generator exceeds the accuracy-matched null by ≥ 0.10 on validated scenarios → **met** (0.16); H3 scenario profile ≥ 0.6 and above every readout → **not met** (0.68, below several readouts).

### Reading

A frozen prediction-trained encoder carries human-like physical expectation: with a linear readout it predicts which clips people get wrong at r ≈ 0.4 where an agent of equal accuracy predicts nothing. A current 2B video generator's imagined outcome carries part of the same signal, zero-shot, above static and kinematic readouts on the scenarios where the measurement is valid, but less than the encoder; its outcome accuracy (0.58) is the bottleneck. Within-scenario readouts use label supervision on other Physion stimuli; the leave-one-scenario-out and generator numbers do not.

## Corrections log

- **Scaling.** Within-scene z-scoring was transductive on covariates. With training-fold-only scaling, the DINOv2 gain is about half the original (+8.9 vs +17.4); both are reported.
- **Wording.** The slowdown duration was first written as "125 ms", taken from the file names; the paper reports 158–190 ms. The boundary-attenuation figure was first described as a share "explained"; it is coefficient attenuation.
