"""Data and results locations.

Data are downloaded by ``scripts/download_data.sh`` and are never committed. Set
``WORLD_MAPS_DATA`` to use a different data directory.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("WORLD_MAPS_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("WORLD_MAPS_RESULTS", REPO / "results"))

# Physical-event study (Yates, Yasuda & Yildirim 2024; OSF ma32p)
EVENTS = DATA / "events"
EVENT_VIDEOS = EVENTS / "Exp1_TemporalProbe_Videos"
EVENT_TRIALS = EVENTS / "Data_Files_CSV" / "trialdata_temporal_dict.csv"
# Lab analysis repository (expert boundaries, clip lengths, pixel changes)
EVENTS_ANALYSIS = DATA / "physical_events_analysis"
BOUNDARIES = EVENTS_ANALYSIS / "boundary_dictionary.json"
LENGTHS = EVENTS_ANALYSIS / "length_dictionary.json"
PIXEL_CHANGES = EVENTS_ANALYSIS / "pixel_changes"

# Woven (Bi et al. 2025; OSF 29jnd, CC0)
WOVEN = DATA / "woven"
