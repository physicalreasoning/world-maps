# Shape under cloth: humans, PbAS and modern models

Humans: 162 workers kept. Unlimited-time accuracy, occluded: all 0.85, same-category 0.73, different-category 0.97; unoccluded same-category 0.87.
Split-half ceiling (unlimited): occluded all 0.80, occluded same-category 0.71, unoccluded same-category 0.80.
PbAS (released runs, iteration 110): occluded same-category acc 0.74, trial r same-category 0.44 [0.32, 0.55], all trials 0.62; unoccluded same-category r 0.44; RT r (log) occluded 0.74, unoccluded 0.81. Paper: 0.44 / 0.44, RT 0.71 / 0.67.

| model | params | best layer | occluded acc same / diff | unoccluded acc same | occluded r same [CI] | occluded r all | unocc r same | margin–RT ρ | de-drape acc same (loo / loco) | de-drape loco r same | imagine drape acc / r same | imagine undrape r same | p vs PbAS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinoB | 87M | L9_grid | 0.35 / 0.53 | 0.48 | 0.16 [0.07, 0.25] | 0.22 | 0.01 | -0.03 | 0.67 / 0.68 | -0.12 | – / – | – | 0.000 |
| dinoS | 22M | L12_grid | 0.48 / 0.63 | 0.58 | 0.12 [0.03, 0.20] | 0.21 | 0.24 | -0.26 | 0.62 / 0.55 | -0.06 | – / – | – | 0.000 |
| siglip2 | 1136M | final_pooled | 0.53 / 0.58 | 0.75 | 0.10 [0.02, 0.18] | 0.19 | 0.41 | -0.35 | 0.57 / 0.62 | -0.10 | – / – | – | 0.000 |
| dinoL | 304M | L20_grid | 0.40 / 0.65 | 0.63 | 0.04 [-0.03, 0.12] | 0.11 | 0.20 | -0.14 | 0.63 / 0.52 | -0.02 | – / – | – | 0.000 |

Readouts: BU = paper protocol on frozen embeddings; de-drape = linear map draped→bare learned on other trials (loo) or other categories (loco); imagine = generator-in-the-loop analysis-by-synthesis (if imagined images were produced). r values are means over participant bootstraps; CI 95%.