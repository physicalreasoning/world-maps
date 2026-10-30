# Physion: generators as simulators vs readouts, against human predictions

Stimuli with human data: 1144. Human P(yes) reliability (binomial-corrected): all 0.99; human accuracy by scenario: collide 0.81, contain 0.76, dominoes 0.69, drape 0.68, drop 0.74, link 0.64, roll 0.87, support 0.76
Detector on real outcomes (rule / calibrated CV accuracy): collide 0.78/0.85, contain 0.66/0.68, dominoes 0.88/0.95, drape 0.74/0.72, drop 0.73/0.77, link 0.58/0.64, roll 0.90/0.89, support 0.90/0.93. Primary scenarios (>= gate): collide, dominoes, drop, roll, support.

## Readouts on all stimuli

| readout | n | accuracy | r with human P(yes) [CI] | κ vs majority | partial r given label [CI] (null) | primary: partial r [CI] |
|---|---|---|---|---|---|---|
| kinematic/within | 1144 | 0.52 | 0.04 [-0.02, 0.09] | 0.02 | 0.03 [-0.02, 0.09] (0.00) | 0.03 [-0.04, 0.11] |
| kinematic/loso | 1144 | 0.49 | -0.04 [-0.10, 0.02] | -0.05 | -0.03 [-0.09, 0.03] (0.00) | -0.03 [-0.11, 0.05] |
| dinoL/mean/within | 1144 | 0.58 | 0.21 [0.15, 0.26] | 0.17 | 0.13 [0.08, 0.19] (0.00) | 0.09 [0.01, 0.16] |
| dinoL/mean/loso | 1144 | 0.54 | 0.12 [0.06, 0.18] | 0.10 | 0.07 [0.01, 0.13] (0.00) | -0.01 [-0.09, 0.06] |
| dinoL/last/within | 1144 | 0.59 | 0.26 [0.21, 0.31] | 0.20 | 0.16 [0.10, 0.22] (0.00) | 0.11 [0.04, 0.19] |
| dinoL/last/loso | 1144 | 0.51 | 0.08 [0.03, 0.14] | 0.03 | 0.06 [0.00, 0.11] (0.00) | 0.02 [-0.05, 0.10] |
| dinoL/last_minus_first/within | 1144 | 0.61 | 0.27 [0.21, 0.32] | 0.24 | 0.18 [0.13, 0.24] (0.00) | 0.21 [0.14, 0.27] |
| dinoL/last_minus_first/loso | 1144 | 0.52 | 0.10 [0.04, 0.15] | 0.10 | 0.12 [0.07, 0.18] (0.00) | 0.15 [0.08, 0.22] |
| siglip2/mean/within | 1144 | 0.58 | 0.22 [0.16, 0.27] | 0.17 | 0.12 [0.06, 0.18] (0.00) | 0.04 [-0.03, 0.12] |
| siglip2/mean/loso | 1144 | 0.53 | 0.12 [0.07, 0.17] | 0.07 | 0.11 [0.06, 0.17] (0.00) | -0.02 [-0.09, 0.05] |
| siglip2/last/within | 1144 | 0.58 | 0.31 [0.26, 0.36] | 0.25 | 0.23 [0.17, 0.28] (0.00) | 0.17 [0.09, 0.24] |
| siglip2/last/loso | 1144 | 0.51 | 0.14 [0.09, 0.19] | 0.12 | 0.14 [0.09, 0.20] (0.00) | -0.04 [-0.11, 0.04] |
| siglip2/last_minus_first/within | 1144 | 0.56 | 0.26 [0.20, 0.31] | 0.19 | 0.21 [0.15, 0.27] (0.00) | 0.24 [0.17, 0.30] |
| siglip2/last_minus_first/loso | 1144 | 0.53 | 0.12 [0.07, 0.18] | 0.10 | 0.13 [0.07, 0.18] (0.00) | 0.16 [0.09, 0.23] |
| vjepaH/mean/within | 1144 | 0.62 | 0.41 [0.36, 0.46] | 0.31 | 0.32 [0.27, 0.37] (0.00) | 0.28 [0.21, 0.35] |
| vjepaH/mean/loso | 1144 | 0.54 | 0.21 [0.15, 0.26] | 0.14 | 0.18 [0.12, 0.23] (0.00) | 0.11 [0.04, 0.19] |
| vjepaH/last_slot/within | 1144 | 0.67 | 0.47 [0.42, 0.51] | 0.34 | 0.33 [0.27, 0.38] (0.00) | 0.27 [0.20, 0.34] |
| vjepaH/last_slot/loso | 1144 | 0.58 | 0.22 [0.17, 0.27] | 0.16 | 0.14 [0.09, 0.20] (0.00) | 0.13 [0.06, 0.21] |
| vjepaH/last_grid/within | 1144 | 0.64 | 0.48 [0.43, 0.52] | 0.37 | 0.37 [0.32, 0.42] (0.00) | 0.34 [0.27, 0.41] |
| vjepaH/last_grid/loso | 1144 | 0.57 | 0.19 [0.14, 0.25] | 0.20 | 0.14 [0.08, 0.20] (0.00) | 0.17 [0.09, 0.24] |
| vjepaL/mean/within | 1144 | 0.61 | 0.37 [0.32, 0.42] | 0.28 | 0.27 [0.22, 0.33] (0.00) | 0.20 [0.13, 0.26] |
| vjepaL/mean/loso | 1144 | 0.57 | 0.20 [0.14, 0.26] | 0.15 | 0.15 [0.10, 0.21] (0.00) | 0.04 [-0.03, 0.11] |
| vjepaL/last_slot/within | 1144 | 0.63 | 0.43 [0.38, 0.48] | 0.34 | 0.29 [0.24, 0.35] (0.00) | 0.23 [0.16, 0.30] |
| vjepaL/last_slot/loso | 1144 | 0.57 | 0.17 [0.11, 0.23] | 0.15 | 0.11 [0.06, 0.17] (0.00) | 0.06 [-0.02, 0.12] |
| vjepaL/last_grid/within | 1144 | 0.62 | 0.41 [0.36, 0.46] | 0.31 | 0.27 [0.22, 0.33] (0.00) | 0.23 [0.15, 0.30] |
| vjepaL/last_grid/loso | 1144 | 0.52 | 0.03 [-0.03, 0.08] | 0.01 | -0.02 [-0.07, 0.05] (0.00) | 0.02 [-0.05, 0.10] |
| vmaeL/mean/within | 1144 | 0.56 | 0.29 [0.23, 0.34] | 0.20 | 0.25 [0.20, 0.31] (0.00) | 0.23 [0.16, 0.30] |
| vmaeL/mean/loso | 1144 | 0.54 | 0.18 [0.12, 0.23] | 0.13 | 0.14 [0.08, 0.20] (0.00) | 0.12 [0.05, 0.19] |
| vmaeL/last_grid/within | 1144 | 0.60 | 0.34 [0.28, 0.39] | 0.23 | 0.26 [0.21, 0.32] (0.00) | 0.22 [0.15, 0.29] |
| vmaeL/last_grid/loso | 1144 | 0.57 | 0.26 [0.21, 0.31] | 0.20 | 0.20 [0.15, 0.26] (0.00) | 0.21 [0.14, 0.28] |
| vlm | 1144 | 0.59 | 0.22 [0.17, 0.28] | 0.17 | 0.12 [0.06, 0.17] (0.00) | 0.21 [0.14, 0.28] |

## Generator: ltx (320 stimuli, 1.0 samples each)

| readout (same subset) | accuracy | r with human P(yes) [CI] | κ | partial r given label [CI] (null) | primary: partial r [CI] |
|---|---|---|---|---|---|
| **ltx p_cal** | 0.58 | 0.17 [0.06, 0.28] | 0.13 | 0.07 [-0.03, 0.17] (-0.00) | 0.16 [0.04, 0.29] |
| **ltx p_rule** | 0.56 | 0.19 [0.09, 0.30] | 0.13 | 0.14 [0.04, 0.25] (-0.00) | 0.11 [-0.03, 0.25] |
| kinematic/within | 0.52 | 0.08 [-0.03, 0.18] | 0.02 | 0.07 [-0.05, 0.19] (-0.00) | 0.03 [-0.13, 0.18] |
| kinematic/loso | 0.49 | -0.00 [-0.10, 0.10] | -0.05 | 0.01 [-0.11, 0.13] (-0.00) | -0.04 [-0.18, 0.11] |
| dinoL/mean/within | 0.59 | 0.18 [0.07, 0.29] | 0.14 | 0.09 [-0.02, 0.20] (-0.00) | -0.03 [-0.15, 0.11] |
| dinoL/mean/loso | 0.55 | 0.13 [0.02, 0.23] | 0.13 | 0.08 [-0.02, 0.18] (-0.00) | -0.07 [-0.20, 0.06] |
| dinoL/last/within | 0.59 | 0.25 [0.15, 0.35] | 0.20 | 0.12 [0.02, 0.23] (-0.00) | 0.00 [-0.13, 0.13] |
| dinoL/last/loso | 0.51 | 0.04 [-0.07, 0.14] | -0.01 | 0.02 [-0.10, 0.12] (-0.00) | -0.07 [-0.20, 0.05] |
| dinoL/last_minus_first/within | 0.62 | 0.27 [0.17, 0.37] | 0.25 | 0.16 [0.05, 0.26] (-0.00) | 0.22 [0.10, 0.34] |
| dinoL/last_minus_first/loso | 0.51 | 0.09 [-0.02, 0.20] | 0.11 | 0.11 [-0.00, 0.21] (-0.00) | 0.20 [0.08, 0.32] |
| siglip2/mean/within | 0.57 | 0.16 [0.05, 0.26] | 0.15 | 0.09 [-0.02, 0.20] (-0.00) | -0.03 [-0.17, 0.10] |
| siglip2/mean/loso | 0.51 | 0.06 [-0.05, 0.17] | 0.05 | 0.09 [-0.02, 0.20] (-0.00) | -0.06 [-0.19, 0.08] |
| siglip2/last/within | 0.56 | 0.30 [0.20, 0.39] | 0.22 | 0.25 [0.15, 0.36] (-0.00) | 0.14 [-0.00, 0.28] |
| siglip2/last/loso | 0.48 | 0.09 [-0.02, 0.20] | 0.11 | 0.14 [0.02, 0.25] (-0.00) | -0.05 [-0.19, 0.11] |
| siglip2/last_minus_first/within | 0.58 | 0.30 [0.19, 0.39] | 0.26 | 0.25 [0.15, 0.35] (-0.00) | 0.28 [0.16, 0.40] |
| siglip2/last_minus_first/loso | 0.51 | 0.13 [0.02, 0.23] | 0.11 | 0.17 [0.06, 0.26] (-0.00) | 0.24 [0.12, 0.36] |
| vjepaH/mean/within | 0.60 | 0.37 [0.27, 0.47] | 0.31 | 0.32 [0.22, 0.42] (-0.00) | 0.19 [0.07, 0.31] |
| vjepaH/mean/loso | 0.51 | 0.20 [0.10, 0.30] | 0.11 | 0.22 [0.12, 0.32] (-0.00) | 0.09 [-0.05, 0.22] |
| vjepaH/last_slot/within | 0.65 | 0.49 [0.40, 0.57] | 0.35 | 0.38 [0.28, 0.48] (-0.00) | 0.25 [0.11, 0.38] |
| vjepaH/last_slot/loso | 0.56 | 0.19 [0.08, 0.29] | 0.13 | 0.16 [0.06, 0.26] (-0.00) | 0.16 [0.01, 0.29] |
| vjepaH/last_grid/within | 0.65 | 0.51 [0.41, 0.59] | 0.44 | 0.42 [0.32, 0.51] (-0.00) | 0.29 [0.15, 0.42] |
| vjepaH/last_grid/loso | 0.55 | 0.20 [0.08, 0.31] | 0.23 | 0.18 [0.07, 0.29] (-0.00) | 0.22 [0.08, 0.35] |
| vjepaL/mean/within | 0.59 | 0.34 [0.24, 0.44] | 0.24 | 0.23 [0.12, 0.34] (-0.00) | 0.06 [-0.09, 0.20] |
| vjepaL/mean/loso | 0.57 | 0.21 [0.10, 0.32] | 0.21 | 0.20 [0.09, 0.31] (-0.00) | 0.02 [-0.13, 0.17] |
| vjepaL/last_slot/within | 0.62 | 0.41 [0.33, 0.51] | 0.34 | 0.28 [0.17, 0.38] (-0.00) | 0.15 [0.00, 0.29] |
| vjepaL/last_slot/loso | 0.57 | 0.21 [0.11, 0.31] | 0.14 | 0.16 [0.05, 0.26] (-0.00) | 0.06 [-0.08, 0.20] |
| vjepaL/last_grid/within | 0.61 | 0.37 [0.27, 0.47] | 0.26 | 0.23 [0.13, 0.33] (-0.00) | 0.12 [-0.01, 0.27] |
| vjepaL/last_grid/loso | 0.50 | 0.02 [-0.09, 0.12] | -0.01 | -0.03 [-0.14, 0.07] (-0.00) | -0.03 [-0.17, 0.11] |
| vmaeL/mean/within | 0.56 | 0.23 [0.11, 0.34] | 0.25 | 0.19 [0.08, 0.29] (-0.00) | 0.18 [0.05, 0.31] |
| vmaeL/mean/loso | 0.51 | 0.07 [-0.03, 0.18] | 0.09 | 0.08 [-0.02, 0.18] (-0.00) | 0.10 [-0.04, 0.23] |
| vmaeL/last_grid/within | 0.59 | 0.33 [0.23, 0.42] | 0.22 | 0.26 [0.15, 0.35] (-0.00) | 0.25 [0.12, 0.38] |
| vmaeL/last_grid/loso | 0.57 | 0.25 [0.15, 0.35] | 0.20 | 0.20 [0.09, 0.30] (-0.00) | 0.29 [0.16, 0.42] |
| vlm | 0.57 | 0.21 [0.09, 0.31] | 0.12 | 0.09 [-0.03, 0.19] (-0.00) | 0.20 [0.07, 0.33] |

Paired comparisons (generator calibrated P(yes) minus readout; stimulus bootstrap):
- ltx vs kinematic/within [all]: partial r diff +0.000 [-0.113, +0.119] (p = 0.966); r diff +0.093 [-0.024, +0.206]
- ltx vs kinematic/within [primary]: partial r diff +0.141 [+0.001, +0.287] (p = 0.050); r diff +0.211 [+0.073, +0.351]
- ltx vs kinematic/loso [all]: partial r diff +0.065 [-0.054, +0.180] (p = 0.274); r diff +0.170 [+0.064, +0.279]
- ltx vs kinematic/loso [primary]: partial r diff +0.202 [+0.070, +0.349] (p = 0.004); r diff +0.273 [+0.140, +0.404]
- ltx vs dinoL/mean/within [all]: partial r diff -0.019 [-0.159, +0.120] (p = 0.786); r diff -0.012 [-0.154, +0.142]
- ltx vs dinoL/mean/within [primary]: partial r diff +0.188 [+0.006, +0.355] (p = 0.046); r diff +0.159 [-0.041, +0.351]
- ltx vs dinoL/mean/loso [all]: partial r diff -0.011 [-0.149, +0.125] (p = 0.870); r diff +0.043 [-0.108, +0.192]
- ltx vs dinoL/mean/loso [primary]: partial r diff +0.237 [+0.056, +0.413] (p = 0.012); r diff +0.255 [+0.070, +0.433]
- ltx vs dinoL/last/within [all]: partial r diff -0.053 [-0.188, +0.083] (p = 0.414); r diff -0.080 [-0.212, +0.056]
- ltx vs dinoL/last/within [primary]: partial r diff +0.162 [+0.003, +0.321] (p = 0.050); r diff +0.085 [-0.101, +0.264]
- ltx vs dinoL/last/loso [all]: partial r diff +0.056 [-0.083, +0.197] (p = 0.414); r diff +0.132 [-0.003, +0.267]
- ltx vs dinoL/last/loso [primary]: partial r diff +0.238 [+0.068, +0.402] (p = 0.004); r diff +0.265 [+0.093, +0.434]
- ltx vs dinoL/last_minus_first/within [all]: partial r diff -0.091 [-0.220, +0.042] (p = 0.186); r diff -0.103 [-0.243, +0.051]
- ltx vs dinoL/last_minus_first/within [primary]: partial r diff -0.056 [-0.214, +0.106] (p = 0.508); r diff +0.002 [-0.186, +0.172]
- ltx vs dinoL/last_minus_first/loso [all]: partial r diff -0.036 [-0.195, +0.127] (p = 0.674); r diff +0.079 [-0.073, +0.241]
- ltx vs dinoL/last_minus_first/loso [primary]: partial r diff -0.036 [-0.227, +0.159] (p = 0.710); r diff +0.089 [-0.081, +0.268]
- ltx vs siglip2/mean/within [all]: partial r diff -0.020 [-0.155, +0.117] (p = 0.774); r diff +0.008 [-0.140, +0.161]
- ltx vs siglip2/mean/within [primary]: partial r diff +0.193 [+0.032, +0.350] (p = 0.018); r diff +0.255 [+0.064, +0.445]
- ltx vs siglip2/mean/loso [all]: partial r diff -0.017 [-0.163, +0.130] (p = 0.834); r diff +0.111 [-0.044, +0.278]
- ltx vs siglip2/mean/loso [primary]: partial r diff +0.220 [+0.038, +0.401] (p = 0.024); r diff +0.365 [+0.166, +0.559]
- ltx vs siglip2/last/within [all]: partial r diff -0.179 [-0.302, -0.049] (p = 0.008); r diff -0.132 [-0.273, +0.006]
- ltx vs siglip2/last/within [primary]: partial r diff +0.031 [-0.139, +0.189] (p = 0.680); r diff +0.092 [-0.089, +0.271]
- ltx vs siglip2/last/loso [all]: partial r diff -0.064 [-0.220, +0.082] (p = 0.396); r diff +0.078 [-0.078, +0.238]
- ltx vs siglip2/last/loso [primary]: partial r diff +0.214 [+0.030, +0.386] (p = 0.022); r diff +0.380 [+0.199, +0.554]
- ltx vs siglip2/last_minus_first/within [all]: partial r diff -0.183 [-0.307, -0.049] (p = 0.006); r diff -0.131 [-0.272, +0.014]
- ltx vs siglip2/last_minus_first/within [primary]: partial r diff -0.119 [-0.273, +0.027] (p = 0.120); r diff -0.067 [-0.249, +0.110]
- ltx vs siglip2/last_minus_first/loso [all]: partial r diff -0.094 [-0.247, +0.060] (p = 0.244); r diff +0.035 [-0.117, +0.191]
- ltx vs siglip2/last_minus_first/loso [primary]: partial r diff -0.077 [-0.241, +0.094] (p = 0.378); r diff +0.079 [-0.099, +0.249]
- ltx vs vjepaH/mean/within [all]: partial r diff -0.254 [-0.382, -0.125] (p = 0.000); r diff -0.205 [-0.345, -0.063]
- ltx vs vjepaH/mean/within [primary]: partial r diff -0.024 [-0.175, +0.123] (p = 0.736); r diff -0.029 [-0.204, +0.146]
- ltx vs vjepaH/mean/loso [all]: partial r diff -0.147 [-0.278, -0.022] (p = 0.036); r diff -0.033 [-0.163, +0.110]
- ltx vs vjepaH/mean/loso [primary]: partial r diff +0.073 [-0.081, +0.245] (p = 0.360); r diff +0.118 [-0.053, +0.293]
- ltx vs vjepaH/last_slot/within [all]: partial r diff -0.304 [-0.428, -0.166] (p = 0.000); r diff -0.322 [-0.450, -0.180]
- ltx vs vjepaH/last_slot/within [primary]: partial r diff -0.088 [-0.238, +0.060] (p = 0.230); r diff -0.142 [-0.305, +0.019]
- ltx vs vjepaH/last_slot/loso [all]: partial r diff -0.091 [-0.204, +0.027] (p = 0.120); r diff -0.022 [-0.149, +0.121]
- ltx vs vjepaH/last_slot/loso [primary]: partial r diff +0.008 [-0.147, +0.164] (p = 0.934); r diff +0.055 [-0.122, +0.231]
- ltx vs vjepaH/last_grid/within [all]: partial r diff -0.353 [-0.479, -0.219] (p = 0.000); r diff -0.340 [-0.476, -0.200]
- ltx vs vjepaH/last_grid/within [primary]: partial r diff -0.131 [-0.293, +0.045] (p = 0.128); r diff -0.152 [-0.320, +0.008]
- ltx vs vjepaH/last_grid/loso [all]: partial r diff -0.111 [-0.231, +0.010] (p = 0.086); r diff -0.031 [-0.175, +0.110]
- ltx vs vjepaH/last_grid/loso [primary]: partial r diff -0.054 [-0.232, +0.137] (p = 0.574); r diff +0.051 [-0.124, +0.231]
- ltx vs vjepaL/mean/within [all]: partial r diff -0.159 [-0.285, -0.036] (p = 0.014); r diff -0.173 [-0.318, -0.037]
- ltx vs vjepaL/mean/within [primary]: partial r diff +0.105 [-0.027, +0.246] (p = 0.138); r diff -0.008 [-0.176, +0.158]
- ltx vs vjepaL/mean/loso [all]: partial r diff -0.127 [-0.273, -0.000] (p = 0.050); r diff -0.045 [-0.202, +0.093]
- ltx vs vjepaL/mean/loso [primary]: partial r diff +0.143 [-0.016, +0.315] (p = 0.084); r diff +0.135 [-0.039, +0.307]
- ltx vs vjepaL/last_slot/within [all]: partial r diff -0.205 [-0.333, -0.072] (p = 0.006); r diff -0.246 [-0.385, -0.111]
- ltx vs vjepaL/last_slot/within [primary]: partial r diff +0.019 [-0.147, +0.183] (p = 0.842); r diff -0.108 [-0.281, +0.073]
- ltx vs vjepaL/last_slot/loso [all]: partial r diff -0.088 [-0.230, +0.048] (p = 0.198); r diff -0.040 [-0.184, +0.095]
- ltx vs vjepaL/last_slot/loso [primary]: partial r diff +0.099 [-0.072, +0.277] (p = 0.280); r diff +0.079 [-0.107, +0.262]
- ltx vs vjepaL/last_grid/within [all]: partial r diff -0.164 [-0.298, -0.027] (p = 0.016); r diff -0.202 [-0.337, -0.052]
- ltx vs vjepaL/last_grid/within [primary]: partial r diff +0.040 [-0.122, +0.202] (p = 0.640); r diff -0.043 [-0.215, +0.131]
- ltx vs vjepaL/last_grid/loso [all]: partial r diff +0.102 [-0.023, +0.221] (p = 0.114); r diff +0.147 [+0.012, +0.280]
- ltx vs vjepaL/last_grid/loso [primary]: partial r diff +0.192 [+0.031, +0.357] (p = 0.014); r diff +0.199 [+0.026, +0.364]
- ltx vs vmaeL/mean/within [all]: partial r diff -0.122 [-0.249, +0.011] (p = 0.060); r diff -0.062 [-0.210, +0.095]
- ltx vs vmaeL/mean/within [primary]: partial r diff -0.015 [-0.169, +0.144] (p = 0.852); r diff +0.019 [-0.152, +0.184]
- ltx vs vmaeL/mean/loso [all]: partial r diff -0.010 [-0.135, +0.116] (p = 0.884); r diff +0.095 [-0.045, +0.235]
- ltx vs vmaeL/mean/loso [primary]: partial r diff +0.067 [-0.080, +0.221] (p = 0.404); r diff +0.156 [-0.007, +0.312]
- ltx vs vmaeL/last_grid/within [all]: partial r diff -0.188 [-0.319, -0.058] (p = 0.004); r diff -0.159 [-0.298, -0.009]
- ltx vs vmaeL/last_grid/within [primary]: partial r diff -0.088 [-0.243, +0.061] (p = 0.266); r diff -0.059 [-0.222, +0.106]
- ltx vs vmaeL/last_grid/loso [all]: partial r diff -0.126 [-0.252, -0.004] (p = 0.044); r diff -0.086 [-0.227, +0.054]
- ltx vs vmaeL/last_grid/loso [primary]: partial r diff -0.124 [-0.277, +0.037] (p = 0.128); r diff -0.053 [-0.215, +0.120]
- ltx vs vlm [all]: partial r diff -0.014 [-0.112, +0.083] (p = 0.806); r diff -0.037 [-0.146, +0.070]
- ltx vs vlm [primary]: partial r diff -0.036 [-0.162, +0.090] (p = 0.600); r diff -0.059 [-0.192, +0.067]

Scenario-profile correlation with human accuracy (Spearman over 8 scenarios): kinematic/within 0.90, kinematic/loso 0.61, dinoL/mean/within -0.26, dinoL/mean/loso 0.23, dinoL/last/within -0.57, dinoL/last/loso 0.29, dinoL/last_minus_first/within 0.07, dinoL/last_minus_first/loso 0.57, siglip2/mean/within -0.62, siglip2/mean/loso -0.45, siglip2/last/within 0.01, siglip2/last/loso -0.57, siglip2/last_minus_first/within 0.31, siglip2/last_minus_first/loso 0.50, vjepaH/mean/within 0.57, vjepaH/mean/loso 0.83, vjepaH/last_slot/within 0.31, vjepaH/last_slot/loso 0.74, vjepaH/last_grid/within 0.40, vjepaH/last_grid/loso 0.71, vjepaL/mean/within -0.14, vjepaL/mean/loso 0.43, vjepaL/last_slot/within 0.52, vjepaL/last_slot/loso 0.73, vjepaL/last_grid/within -0.02, vjepaL/last_grid/loso 0.86, vmaeL/mean/within -0.12, vmaeL/mean/loso 0.48, vmaeL/last_grid/within 0.26, vmaeL/last_grid/loso 0.50, vlm 0.86

partial r given label: correlation between model and human P(yes) after the ground-truth label is regressed out of both, i.e. agreement on *which* stimuli are judged wrongly. null = accuracy-matched random agent. Primary = scenarios where the contact detector passed the validity gate on real outcomes.