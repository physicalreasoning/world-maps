# World Maps

[![Tests](https://github.com/physicalreasoning/world-maps/actions/workflows/tests.yml/badge.svg)](https://github.com/physicalreasoning/world-maps/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Do video models build world maps? Testing video-model internals against human physical perception**

[[Findings]](FINDINGS.md)
[[Data and licenses]](docs/DATA.md)
[[Reproduction]](docs/REPRODUCE.md)

## Status

**Exploratory pilot on public data.** Two analyses are complete. Both use human data publicly released by the Yale Cognitive and Neural Computation Lab (CNCL). Confirmatory studies are planned and not yet run. This is not an official lab release.

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
src/world_maps/               paths and tested helpers
results/                      committed outputs (features and JSON results)
tests/                        unit tests (run in CI)
```

## Quick start

```bash
uv sync --extra models --no-config --default-index https://pypi.org/simple
bash scripts/download_data.sh                 # about 270 MB
uv run python scripts/analyze_events.py       # uses committed results/events_features.csv
```

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
