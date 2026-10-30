| Signal | Family | Size | Params (M) | Boundary AUROC [95% CI] | Held-out gain ×10⁻³ nats/trial [95% CI] | Holm p |
|---|---|---|---|---|---|---|
| dino_change | DINOv2 | base | 86 | 0.79 [0.72, 0.85] | +17.4 [+8.9, +25.8] | nan |
| dinoL_change | DINOv2 | L | 304 | 0.82 [0.76, 0.87] | +20.9 [+10.5, +30.8] | 0.005 |
| dinog_change | DINOv2 | g | 1136 | 0.82 [0.76, 0.87] | +19.3 [+9.2, +29.2] | 0.005 |
| vlm_boundary | Qwen3-VL | so400m | 8767 | 0.74 [0.69, 0.78] | +2.8 [-1.1, +6.6] | 1.000 |
| vlm_unpredictable | Qwen3-VL | so400m | 8767 | 0.49 [0.41, 0.57] | -0.4 [-0.9, +0.1] | 1.000 |
| siglip2_change | SigLIP 2 | so400m | 1136 | 0.80 [0.74, 0.85] | +10.8 [+3.3, +18.7] | 0.016 |
| vjepa_surprise | V-JEPA 2 | L | 300 | 0.65 [0.59, 0.71] | +0.5 [-1.6, +2.5] | nan |
| vjepaH_surprise | V-JEPA 2 | H | 654 | 0.63 [0.56, 0.69] | +1.7 [-0.9, +4.2] | 1.000 |
| vjepaH_change | V-JEPA 2 | H | 654 | 0.47 [0.40, 0.55] | +0.6 [-1.7, +2.8] | 1.000 |
| vjepag_surprise | V-JEPA 2 | g | 1035 | 0.62 [0.54, 0.69] | +0.4 [-1.1, +1.7] | 1.000 |
| vjepag_change | V-JEPA 2 | g | 1035 | 0.58 [0.52, 0.65] | +0.4 [-1.1, +1.7] | 1.000 |
| vmae_surprise | VideoMAE | base | 87 | 0.41 [0.37, 0.46] | -0.2 [-1.6, +1.0] | nan |
| vmaeL_surprise | VideoMAE | L | 343 | 0.42 [0.36, 0.48] | +0.1 [-1.0, +1.2] | 1.000 |

1509 bins, 32 scenes, 6239 probe trials. flow_accel boundary AUROC 0.69. Pilot 0 rows (L/base) are reference values on the same bins; Holm is over the new signals only.
