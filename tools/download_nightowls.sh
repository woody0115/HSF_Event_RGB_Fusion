#!/usr/bin/env bash
# Download the NightOwls night-pedestrian dataset (Neumann et al., ACCV 2018; non-commercial research use,
# https://www.nightowls-dataset.org/) for extra RGB-teacher training. Training images + labels, and the
# validation labels only (validation images are 50 GB and we decide on DSEC val anyway). Test labels are not public.
# The training zip (142.3 GB) is kept after extraction; nothing is deleted.
# Also fetches the PEOD README and terms of use from GitHub, to check its classes before any bulk download.
# Layout:  $ROOT/NightOwls/{nightowls_training.zip, nightowls_training.json, nightowls_validation.json, training/...}
#          $ROOT/PEOD/docs/{README.md, Term_of_Use_and_License.md}
# Usage:   setsid nohup bash tools/download_nightowls.sh > ~/work/shared_data/NightOwls/download.out 2>&1 &
set -u

ROOT=${1:-$HOME/work/shared_data}
BASE=https://thor.robots.ox.ac.uk/nightowls/python
DIR=$ROOT/NightOwls
LOG=$DIR/download.log
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DIR" "$ROOT/PEOD/docs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }
log start

# 1) small files
for f in nightowls_training.json nightowls_validation.json nightowls_test_imageids.json; do
  [ -s "$DIR/$f" ] || curl -sSLf --retry 5 -o "$DIR/$f" "$BASE/$f" || { log "FAILED: $f"; exit 1; }
  log "got $f ($(du -h "$DIR/$f" | cut -f1))"
done
PEOD=https://raw.githubusercontent.com/bupt-ai-cz/PEOD/main
curl -sSLf -o "$ROOT/PEOD/docs/README.md" "$PEOD/README.md" && log "got PEOD README" || log "PEOD README failed"
curl -sSLf -o "$ROOT/PEOD/docs/Term_of_Use_and_License.md" "$PEOD/Term%20of%20Use%20and%20License.md" \
  && log "got PEOD terms" || log "PEOD terms failed"

# 2) training images: resumable, parallel range requests; fall back to wget -c if ranges are refused
ZIP=$DIR/nightowls_training.zip
if [ ! -f "$ZIP.done" ]; then
  log "download nightowls_training.zip (142.3 GB)"
  ok=0
  for try in 1 2 3 4 5; do
    if "$PYTHON" "$HERE/pdownload.py" "$BASE/nightowls_training.zip" "$ZIP" --workers 4; then ok=1; break; fi
    log "  pdownload failed, retry $try"; sleep 30
  done
  if [ $ok -ne 1 ]; then
    log "  falling back to wget -c"
    wget -c -q --tries=20 -O "$ZIP" "$BASE/nightowls_training.zip" && ok=1
  fi
  [ $ok -eq 1 ] || { log "FAILED download"; exit 1; }
  [ "$(stat -c %s "$ZIP")" = 142270733872 ] || { log "FAILED size check: $(stat -c %s "$ZIP")"; exit 1; }
  touch "$ZIP.done"
fi
log "zip ok: $(du -h "$ZIP" | cut -f1)"

# 3) extract (zip kept)
if [ ! -f "$DIR/extract.done" ]; then
  log "extract; first entries: $(unzip -Z1 "$ZIP" | head -3 | tr '\n' ' ')"
  unzip -q -n "$ZIP" -d "$DIR" || { log "FAILED extract"; exit 1; }
  touch "$DIR/extract.done"
fi
log "png count: $(find "$DIR" -name '*.png' | wc -l); free: $(df -h "$ROOT" | awk 'NR==2 {print $4}')"
log 'ALL DONE'
