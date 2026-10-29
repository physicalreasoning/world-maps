# Data and licenses

Nothing in `data/` is committed. `scripts/download_data.sh` fetches everything from the original sources.

| Source | Contents used | Location | License |
|---|---|---|---|
| Yates, Yasuda & Yildirim (2024), Experiment 1 | 32 base videos and slowdown-probe videos; trial-level responses (`trialdata_temporal_dict.csv`) | OSF [ma32p](https://osf.io/ma32p/): `Exp1_TemporalProbe_Videos.zip`, `Data_Files_CSV.zip` | None stated; downloaded, not redistributed |
| CNCL `physical_events_analysis` | Expert boundaries, clip lengths, pixel changes | [GitHub](https://github.com/CNCLgithub/physical_events_analysis), commit `3f2a4ae` | None stated; cloned, not redistributed |
| Bi et al. (2025), Woven | Trial-level human choices with per-trial Woven, ablation, and DNN responses | OSF [29jnd](https://osf.io/29jnd/): `parsed_questions_stiff.csv`, `parsed_questions_mass.csv` | CC0 1.0 |
| Yildirim et al. (2024), shape under cloth | 720 stimulus images (120 trials × 2 occlusion conditions × 3 images) | [GitHub](https://github.com/CNCLgithub/intuitive-physics-3d-shape-perception-stimuli), cloned on first run | None stated; cloned, not redistributed |
| Yildirim et al. (2024), behaviour and PbAS runs | `processed_data_anonymous.csv` (20,880 trials), `pbas-full-model-anp.npy`, `td-no-image-encoding-anp.npy` | [GitHub](https://github.com/CNCLgithub/PbAS-model-human-comparisons), cloned on first run | None stated; cloned, not redistributed |
| Bear et al. (2021), Physion test set | `PhysionTest-Core`: 1,200 videos (red/yellow renders, human-facing renders, first-frame maps) | [S3 zip](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip) (270 MB), downloaded on first run | MIT (repository licence) |
| Bear et al. (2021), labels and human data | `stimulus_names_and_labels.csv`, `results/csv/humans/*.csv` (per-stimulus accuracy and participant counts) | [GitHub](https://github.com/cogtoolslab/physics-benchmarking-neurips2021), cloned on first run | MIT |
| Model weights | `facebook/vjepa2-vitl-fpc64-256`, `MCG-NJU/videomae-base`, `facebook/dinov2-*`, `google/siglip2-so400m-patch16-256`, `openai/clip-vit-large-patch14`, `stabilityai/stable-diffusion-xl-base-1.0`, `Lightricks/LTX-Video*` (imagination and continuation stages) | Hugging Face, downloaded on first run | Per model card |

Base videos: for each scene we use the original 120 fps render whose probe index lies beyond the clip end, so no slowdown is shown. For one scene (`falling_falling4316`), only countdown-prefixed 30 fps renders exist; the countdown is stripped automatically.

The Woven stimulus videos are not in either public archive. The download link documented in the lab's experiment repository returned HTTP 404 on October 23, 2026.
