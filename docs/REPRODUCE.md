# Reproduction

```bash
uv sync --extra models --no-config --default-index https://pypi.org/simple
bash scripts/download_data.sh

# 1. Model signals (about 46 min on an Apple M4 Pro; skip to reuse results/events_features.csv)
uv run python scripts/extract_event_signals.py          # -> results/events_features.csv
uv run python scripts/extract_event_signals.py --limit 1 --out /tmp/smoke.csv   # smoke test

# 2. Prespecified analysis                                # -> results/events_results.json
uv run python scripts/analyze_events.py

# 3. Sensitivity analyses and exploratory breakdown
uv run python scripts/event_sensitivity.py              # -> results/events_sensitivity.json
uv run python scripts/event_type_breakdown.py           # -> results/events_exploratory.json

# 4. Probe timing check                                   # -> results/events_alignment.json
uv run python scripts/check_event_alignment.py

# 5. Woven reanalysis                                     # -> results/woven_agree.json
uv run python scripts/woven_reanalysis.py agree
# 6. Shape under cloth (stimuli and behaviour are cloned on first run)
uv run python scripts/cloth_extract.py --models dinoS dinoB dinoL siglip2   # -> results/cloth/features/*.npz (not committed)
uv run python scripts/cloth_analyze.py                                      # -> results/cloth/cloth_results.{json,md}, figures/
# GPU only: imagination stage, then re-embed and re-analyse
uv run python scripts/cloth_imagine.py --generator sdxl --k 4
uv run python scripts/cloth_extract.py --models dinoB dinoL siglip2 && uv run python scripts/cloth_analyze.py
# Colab: notebooks/colab_draped_shapes.ipynb (regenerate with python notebooks/build_cloth_notebook.py)
```

All analysis scripts use fixed random seeds. On MPS, model signals may differ from CUDA in the last decimal places.
