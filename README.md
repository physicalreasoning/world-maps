# World Maps

[![Tests](https://github.com/physicalreasoning/world-maps/actions/workflows/tests.yml/badge.svg)](https://github.com/physicalreasoning/world-maps/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Do video models build world maps? Testing video-model internals against human physical perception**

[[Findings]](FINDINGS.md)
[[Data and licenses]](docs/DATA.md)
[[Reproduction]](docs/REPRODUCE.md)

## Status

**Exploratory studies on public data.** Five analyses are complete (physical events at two scales, Woven, shape under cloth, Physion). Both use human data publicly released by the Yale Cognitive and Neural Computation Lab (CNCL). Confirmatory studies are planned and not yet run. This is not an official lab release.

## The question

Most evaluations of physics in video models ask whether a model *notices* an impossible event, or whether generated videos *look* plausible. CNCL's work proposes that human physical perception runs on approximate, structure-preserving representations used for inference and prediction.

We ask whether video models carry physical structure of that kind. **And does it explain where humans succeed and fail?** The lab's paradigms make this testable: cloth, liquids, physical events, and tracking, with trial-level human data and process models that already predict human errors.

## Results at a glance

### 1. Physical events: do video models break events where humans do?

Data: [Yates, Yasuda & Yildirim (2024)](https://cncl.yale.edu/assets/pdf/2024_YatesetalEvents.pdf). Participants watched 32 videos of collisions, falls, containment, and occlusion, and pressed a key when they noticed a brief slowdown.

- **The published effect reproduces exactly from the public data.** Slowdowns at event boundaries are detected 33.2% of the time vs 61.1% elsewhere (109 participants; t(107) = −11.36).
- **V-JEPA 2's prediction error marks event boundaries** (AUROC 0.65 [0.59, 0.71]). The model sees only the past. Its marking is no better than raw motion acceleration (0.69), and it shows no detectable gain beyond low-level motion in predicting where people miss slowdowns.
- **A static image model did better.** DINOv2's frame-to-frame change predicted held-out human detection better than V-JEPA 2's error: +10.0 × 10⁻³ nats per trial, 95% CI [0.5, 19.1] under crossed participant × scene resampling. With whole event-pair families held out, the interval includes zero ([−4.0, 19.5]), so this is **exploratory**.
- **Descriptively**, DINOv2 change marks containment and occlusion boundaries (AUROC 0.90) but not collisions (0.55), where motion acceleration is highest (0.74).

| Signal | Boundary AUROC | Held-out gain in predicting detection (×10⁻³ nats/trial) |
|---|---|---|
| V-JEPA 2-L prediction error (prespecified primary) | 0.65 [0.59, 0.71] | +0.5 [−1.5, +2.4] |
| VideoMAE-base reconstruction error | 0.41 [0.36, 0.46] | −0.2 [−1.5, +0.9] |
| DINOv2-base frame change | 0.79 [0.72, 0.85] | +17.4 [+9.1, +26.4]; +8.9 [+1.8, +15.4] with training-fold scaling |
| Optical-flow acceleration | 0.69 [0.63, 0.75] | (baseline covariate) |

The full table, sensitivity analyses, alignment check, and limits are in [FINDINGS.md](FINDINGS.md).

### 2. Woven (cloth): confirming the lab's model advantage

From the released trials of [Bi et al. (2025)](https://www.nature.com/articles/s41467-025-61458-x) (100 participants per task), Woven agrees with human choices more often than the lab's task-trained network:
- **stiffness:** 12.0 points more (64.4% vs 52.4%);
- **mass:** 14.1 points more (62.8% vs 48.6%).

A normal-approximation sensitivity estimate puts the resolvable difference in this four-scene design at about 8 points.

### 3. Shape under cloth: have foundation models learned to see through the drape? No.

Data: [Yildirim et al. (2024)](https://www.nature.com/articles/s41562-023-01759-7). People match a cloth-draped object to its bare counterpart; the lab's physics-based analysis-by-synthesis model (PbAS) explains trial-level accuracy and response times; ImageNet CNNs sit at chance on the hard (same-category) trials. Stimuli, human data and PbAS runs are public, and our pipeline reproduces the paper's numbers (PbAS r = 0.44 [0.33, 0.54]; ceiling 0.71; RT r 0.74 / 0.81 vs 0.71 / 0.67).

- **The gap is not closed.** With the paper's protocol, the best trial-level correlation with human accuracy on occluded same-category trials is 0.18 [0.11, 0.25] (V-JEPA 2-L); every model's upper bound is below PbAS's lower bound. AlexNet reproduces the paper's chance result (0.53, r = 0.08). CLIP-L and V-JEPA 2 reach 0.60–0.65 accuracy on the hard cell, above chance, with errors that are not human-like.
- **Decodable but not used.** A linear map from the draped target's embedding to bare-object embeddings, learned on held-out categories, lifts hard-cell accuracy to 0.62–0.67 (DINOv2-B/g, V-JEPA 2-H) while its correlation with human difficulty stays near zero.
- **Appearance editing is not simulation.** SDXL in the analysis-by-synthesis loop (drape each candidate, un-drape the target) stays at chance; the generator wraps texture around the object rather than draping a cloth over it. The video-generation route is built; its first run produced no clips.
- **Margin and RT.** Larger decision margins go with faster human responses for V-JEPA 2-H (ρ = −0.41), SigLIP 2 (−0.35), AlexNet (−0.34), DINOv2-g (−0.32); not for the others. Suggestive.

Table, figures and the imagined contact sheet: [results/cloth](results/cloth/cloth_results.md).

### 4. Generators as simulators on Physion: people and V-JEPA 2 get the same clips wrong

Data: [Physion](https://github.com/cogtoolslab/physics-benchmarking-neurips2021) (Bear et al. 2021), 1,144 test clips with per-stimulus human predictions (66–96 participants each; reliability 0.99). People saw 1.5 s and predicted whether the red object would touch the yellow area. Key metric: the partial correlation between model and human P(yes) after the ground-truth label is regressed out of both, i.e. agreement on *which* clips are judged wrongly; accuracy-matched random agents score 0.00.

- **A frozen V-JEPA 2-H with a linear readout predicts human error patterns:** partial r = 0.37 [0.32, 0.42], κ with the human majority 0.37, accuracy 0.64 (humans 0.74). Trained on the other seven scenarios it still transfers (0.14 [0.08, 0.20]). Static encoders carry less (SigLIP 2 0.23, DINOv2-L 0.18), the VLM 0.12, a kinematic heuristic 0.03.
- **A video generator's imagined outcome is human-like, but less so.** LTX-Video 2B continues the 1.5 s context zero-shot; a contact detector validated on the real outcomes (0.81; five scenarios pass the ≥ 0.75 gate) reads the imagined result. On the validated scenarios the imagined contact agrees with human errors at 0.16 [0.04, 0.29], above the static and kinematic readouts (paired lower bounds > 0) and below V-JEPA 2 (−0.13 [−0.29, +0.05]). Accuracy 0.58; the bottleneck is continuation quality.

Table, paired comparisons, clips: [results/physion](results/physion/physion_results.md).

### 5. Physical events at scale: the Pilot 0 dissociation holds

Running the Pilot 0 protocol on larger models: DINOv2-L and DINOv2-g frame change predict where people miss slowdowns (+20.9 and +19.3 × 10⁻³ nats per trial, Holm p = .005; SigLIP 2 +10.8, p = .016), while V-JEPA 2's prediction error adds nothing at L, H or g (≤ +1.7, n.s.) and VideoMAE-L adds nothing. Qwen3-VL's boundary judgment marks boundaries (AUROC 0.74) without predicting detection. [results/events_results_scaled.md](results/events_results_scaled.md).

## Repository layout

```text
scripts/
  download_data.sh            fetch public data into data/ (never committed)
  extract_event_signals.py    per-200 ms signals: V-JEPA 2, VideoMAE, DINOv2, optical flow
  analyze_events.py           prespecified analysis: reproduction, boundary AUROC, held-out prediction
  event_sensitivity.py        review-requested sensitivity analyses
  event_type_breakdown.py     exploratory event-type breakdown
  check_event_alignment.py    verifies probe timing against the original videos
  woven_reanalysis.py         Woven vs DNN agreement with human choices
  extract_event_signals_scaled.py, analyze_events_scaled.py   Pilot 0 protocol on larger models (Colab)
  cloth_extract.py            shape under cloth: embeddings of the 720 stimulus (and imagined) images
  cloth_analyze.py            humans, PbAS reference, model readouts, de-draping, imagination, VLM
  cloth_imagine.py            generator-in-the-loop stage (SDXL, FLUX Kontext, LTX-Video, Wan 2.2; GPU)
  physion_extract.py          Physion: contact-detector validation, kinematic features, encoder features, VLM
  physion_generate.py         Physion: video-generator continuations scored with the detector (GPU)
  physion_analyze.py          Physion: humans, detector gate, readouts, generator, paired comparisons
notebooks/                    Colab notebooks (generated; see build_cloth_notebook.py)
src/world_maps/               paths, tested helpers, and the shape-under-cloth data module
results/                      committed outputs (features and JSON results)
tests/                        unit tests (run in CI)
```

## Quick start

```bash
uv sync --extra models --no-config --default-index https://pypi.org/simple
bash scripts/download_data.sh                 # about 270 MB
uv run python scripts/analyze_events.py       # uses committed results/events_features.csv
```

For the shape-under-cloth study: `uv run python scripts/cloth_extract.py --models dinoB dinoL siglip2` then `uv run python scripts/cloth_analyze.py` (the stimuli and behaviour repositories are cloned on first run).

To recompute the model signals from the videos, run `extract_event_signals.py`. It takes about 46 minutes on an Apple M4 Pro, or a few minutes on a GPU. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full sequence.

## Method notes

- **Leak-free prediction error.** At each 200 ms probe position, V-JEPA 2's context encoder sees only the preceding 0.8 s, sampled at 15 fps. The predictor forecasts the next 0.27 s, and the forecast is compared with the encoding of the real clip. Future frames never enter the context.
- **Time alignment.** The slowdown in probe *k* begins at 0.2*k* s of the original video. A frame-matching check found 58 of 61 probes within 0.1 s of that onset.
- **Prespecified, then stress-tested.** The primary signal, covariates, and tests were fixed before any model signal was compared with human data. After review, we added training-fold-only scaling, event-family holdout, crossed resampling, and exclusions. The conservative numbers are reported alongside the originals.

## Limitations

These results come from 32 videos, a single V-JEPA 2 configuration (15 fps, final layer), and readouts with different temporal windows. Our conclusions concern these readouts, not the architectures themselves. Intervals are conditional on fitted models. The mechanism is open: changes in what is visible are one candidate explanation, and anticipatory updating is another.

## Data and licenses

Code is Apache-2.0. Data are downloaded from their original sources and are not redistributed here; see [docs/DATA.md](docs/DATA.md). The Woven trial data are CC0. The event videos and the lab's analysis repository carry no stated license.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the original studies:

- Yates, T. S., Yasuda, S., & Yildirim, I. (2024). Temporal segmentation and "look ahead" simulation: Physical events structure visual perception of intuitive physics. *Journal of Experimental Psychology: Human Perception and Performance*.
- Bi, W., Shah, A. D., Wong, K. W., Scholl, B. J., & Yildirim, I. (2025). Computational models reveal that intuitive physics underlies visual processing of soft objects. *Nature Communications*.
- Yildirim, I., Siegel, M. H., Soltani, A. A., Ray Chaudhuri, S., & Tenenbaum, J. B. (2024). Perception of 3D shape integrates intuitive physics and analysis-by-synthesis. *Nature Human Behaviour*, 8, 320–335.
- Bear, D. M., et al. (2021). Physion: Evaluating physical prediction from vision in humans and machines. *NeurIPS Datasets and Benchmarks*.
