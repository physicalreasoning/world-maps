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

## Corrections log

- **Scaling.** Within-scene z-scoring was transductive on covariates. With training-fold-only scaling, the DINOv2 gain is about half the original (+8.9 vs +17.4); both are reported.
- **Wording.** The slowdown duration was first written as "125 ms", taken from the file names; the paper reports 158–190 ms. The boundary-attenuation figure was first described as a share "explained"; it is coefficient attenuation.
