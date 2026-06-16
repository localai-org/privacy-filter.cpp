#!/usr/bin/env bash
# Build a privacy-filter "redaction race" demo video end to end: render the
# pii_duel scrolling TUI for a scene, record it with recorder-for-agents, trim
# the dead lead-in, append a branding outro.
#
#   SCENE=cpu ./make.sh            # 8k-token doc, CPU, real time, both finish
#   SCENE=gpu DILATE=4 ./make.sh   # 132k-token doc, GPU, HF OOMs
#
# env: RECORDER, SCENE(cpu|gpu), DILATE, WIDTH/HEIGHT/FPS/FONTSIZE, LINK
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECORDER=${RECORDER:-/home/rich/python/recorder-for-agents}
SCENE=${SCENE:-cpu}
OUT=${1:-pii_duel_${SCENE}.mp4}
DILATE=${DILATE:-1}
W=${WIDTH:-1280}; H=${HEIGHT:-720}; FS=${FONTSIZE:-16}; FPS=${FPS:-30}
LINK=${LINK:-github.com/richiejp/privacy-filter.cpp}
SDIR="$HERE/traces/$SCENE"

[ -d "$SDIR" ] || { echo "no scene at $SDIR (run gen_corpus.py)"; exit 1; }
[ -x "$RECORDER/record.sh" ] || { echo "recorder not found at $RECORDER"; exit 1; }

# capture length: start delay + slowest engine (scaled) + settle + slice of card
DUR=$(python3 - "$SDIR" "$DILATE" <<'PY'
import json, sys, math
from pathlib import Path
d = Path(sys.argv[1]); dil = float(sys.argv[2])
content = json.load(open(d / "content.json")); eng = json.load(open(d / "engines.json"))
n = content["n_tokens"]
slow = max((e.get("oom_at_tokens") or n) / e["tps"] for e in eng)
print(int(math.ceil(1.0 + slow * dil + 1.4 + 0.7 + 2.0)))
PY
)
echo "[make] scene=$SCENE ${W}x${H}@${FPS} fs=${FS} dilate=${DILATE} duration=${DUR}s -> out/$OUT"

WORK="$HERE" BG="#0d1117" FG="#d7dde5" FONTSIZE="$FS" DURATION="$DUR" \
  WIDTH="$W" HEIGHT="$H" FPS="$FPS" START_DELAY=1.0 END_HOLD=0.2 \
  "$RECORDER/record.sh" "python3 pii_duel.py --scene traces/$SCENE --dilate $DILATE --link '$LINK'" "$OUT"

RAW="$HERE/out/$OUT"; NOEXT="${OUT%.mp4}"
if [ -f "$RECORDER/examples/duel/trim_lead.sh" ]; then
  bash "$RECORDER/examples/duel/trim_lead.sh" "$RAW" "$HERE/out/.trim_$SCENE.mp4" \
    && mv "$HERE/out/.trim_$SCENE.mp4" "$RAW"
fi
if [ -f "$RECORDER/examples/duel/outro.sh" ]; then
  OW="$W" OH="$H" TITLE="privacy-filter.cpp" \
  LINK1="github.com/richiejp/privacy-filter.cpp" \
  LINK2="real NER spans · stock ggml · see README Bench" \
    bash "$RECORDER/examples/duel/outro.sh" "$RAW" "$HERE/out/${NOEXT}_final.mp4"
  echo "-> $HERE/out/${NOEXT}_final.mp4"
else
  echo "-> $RAW"
fi
