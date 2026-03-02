#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="assets/screenshots"
mkdir -p "$OUT_DIR"

BASE_URL="http://localhost:8501"

# Adjust browser app name if needed: "Google Chrome" or "Safari"
BROWSER_APP="Google Chrome"

open_scenario() {
  local scenario="$1"
  local encoded
  encoded=$(python3 - <<PY
import urllib.parse
print(urllib.parse.quote("$scenario"))
PY
)
  local url="$BASE_URL/?scenario=$encoded"
  osascript <<OSA
 tell application "$BROWSER_APP"
   activate
   open location "$url"
 end tell
OSA
}

capture() {
  local name="$1"
  # Full-screen capture. Replace with `-l <window_id>` if you prefer window-only.
  screencapture -x "$OUT_DIR/${name}.png"
  echo "Saved: $OUT_DIR/${name}.png"
}

if ! curl -fsS "$BASE_URL/_stcore/health" >/dev/null; then
  echo "Streamlit is not running at $BASE_URL"
  echo "Start it first: streamlit run dashboard/app.py"
  exit 1
fi

scenarios=(
  "Executive Demo"
  "Growth Inbound"
  "Cold Prospect"
  "Large Slow Account"
)

for scenario in "${scenarios[@]}"; do
  echo "Opening scenario: $scenario"
  open_scenario "$scenario"
  sleep 4
  safe_name=$(echo "$scenario" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd 'a-z0-9_')
  capture "$safe_name"
done

echo "All screenshots captured in $OUT_DIR"
