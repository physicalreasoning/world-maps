# Physion: generators as simulators vs readouts, against human predictions

Stimuli with human data: 1144. Human P(yes) reliability (binomial-corrected): all 0.99; human accuracy by scenario: collide 0.81, contain 0.76, dominoes 0.69, drape 0.68, drop 0.74, link 0.64, roll 0.87, support 0.76
Detector on real outcomes (rule / calibrated CV accuracy): collide 0.78/0.85, contain 0.66/0.68, dominoes 0.88/0.95, drape 0.74/0.72, drop 0.73/0.77, link 0.58/0.64, roll 0.90/0.89, support 0.90/0.93. Primary scenarios (>= gate): collide, dominoes, drop, roll, support.

## Readouts on all stimuli

| readout | n | accuracy | r with human P(yes) [CI] | κ vs majority | partial r given label [CI] (null) | primary: partial r [CI] |
|---|---|---|---|---|---|---|
| kinematic/within | 1144 | 0.52 | 0.04 [-0.01, 0.09] | 0.02 | 0.03 [-0.02, 0.08] (0.00) | 0.03 [-0.05, 0.10] |
| kinematic/loso | 1144 | 0.49 | -0.04 [-0.10, 0.02] | -0.05 | -0.03 [-0.09, 0.02] (-0.00) | -0.03 [-0.11, 0.04] |

Scenario-profile correlation with human accuracy (Spearman over 8 scenarios): kinematic/within 0.90, kinematic/loso 0.61

partial r given label: correlation between model and human P(yes) after the ground-truth label is regressed out of both, i.e. agreement on *which* stimuli are judged wrongly. null = accuracy-matched random agent. Primary = scenarios where the contact detector passed the validity gate on real outcomes.