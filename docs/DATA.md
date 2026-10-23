# Data and licenses

Nothing in `data/` is committed. `scripts/download_data.sh` fetches everything from the original sources.

| Source | Contents used | Location | License |
|---|---|---|---|
| Yates, Yasuda & Yildirim (2024), Experiment 1 | 32 base videos and slowdown-probe videos; trial-level responses (`trialdata_temporal_dict.csv`) | OSF [ma32p](https://osf.io/ma32p/): `Exp1_TemporalProbe_Videos.zip`, `Data_Files_CSV.zip` | None stated; downloaded, not redistributed |
| CNCL `physical_events_analysis` | Expert boundaries, clip lengths, pixel changes | [GitHub](https://github.com/CNCLgithub/physical_events_analysis), commit `3f2a4ae` | None stated; cloned, not redistributed |
| Bi et al. (2025), Woven | Trial-level human choices with per-trial Woven, ablation, and DNN responses | OSF [29jnd](https://osf.io/29jnd/): `parsed_questions_stiff.csv`, `parsed_questions_mass.csv` | CC0 1.0 |
| Model weights | `facebook/vjepa2-vitl-fpc64-256`, `MCG-NJU/videomae-base`, `facebook/dinov2-base` | Hugging Face, downloaded on first run | Per model card |

Base videos: for each scene we use the original 120 fps render whose probe index lies beyond the clip end, so no slowdown is shown. For one scene (`falling_falling4316`), only countdown-prefixed 30 fps renders exist; the countdown is stripped automatically.

The Woven stimulus videos are not in either public archive. The download link documented in the lab's experiment repository returned HTTP 404 on October 23, 2026.
