#!/usr/bin/env bash
# Download the public data used here into ./data (or $WORLD_MAPS_DATA).
# Nothing in data/ is committed: the event videos and the lab's analysis repository carry
# no stated license, so we fetch them from the original sources instead of redistributing.
set -euo pipefail
DATA="${WORLD_MAPS_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA/events" "$DATA/woven"

# 1. Physical-event study (Yates, Yasuda & Yildirim 2024), OSF project ma32p
cd "$DATA/events"
for pair in "637929e992a98a11bb06dbc5 Exp1_TemporalProbe_Videos.zip" \
            "64874de0c86116007525260c Data_Files_CSV.zip"; do
  set -- $pair
  [ -f "$2" ] || curl -fL --retry 3 "https://osf.io/download/$1/" -o "$2"
  unzip -oq "$2" -x "__MACOSX/*"
done

# 2. Lab analysis repository: expert boundaries, clip lengths, pixel changes (pinned commit)
if [ ! -d "$DATA/physical_events_analysis" ]; then
  git clone -q https://github.com/CNCLgithub/physical_events_analysis.git "$DATA/physical_events_analysis"
fi
git -C "$DATA/physical_events_analysis" checkout -q 3f2a4aec9f85044ea4ef50edbc6ee41934a40541

# 3. Woven trial data (Bi et al. 2025), OSF project 29jnd, CC0 1.0
cd "$DATA/woven"
[ -f parsed_questions_stiff.csv ] || curl -fL --retry 3 https://osf.io/download/27skd -o parsed_questions_stiff.csv
[ -f parsed_questions_mass.csv ] || curl -fL --retry 3 https://osf.io/download/vyr5n -o parsed_questions_mass.csv

echo "Data ready in $DATA"
