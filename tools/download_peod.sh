#!/usr/bin/env bash
# Download PEOD (Cui et al., arXiv 2511.08140; pixel-aligned event + RGB detection, 6 classes; non-commercial
# research use, see Term of Use) from the authors' public Google Drive folder
# https://drive.google.com/drive/folders/19W2EJUXXtVZA24_PuAQN_n275QjpuV89 . No extra package: file ids were read
# from the folder listing, and drive.usercontent.google.com serves range requests, so tools/pdownload.py works.
# PEOD is a second benchmark to be trained and reported separately from DSEC (2026-11-12 lab meeting).
# Archives are kept after extraction; nothing is deleted.
# Layout:  $ROOT/PEOD/_archives/*   ->  $ROOT/PEOD/{rgb,event,annotations(coco_0111_last),timestamp}/...
# Usage:   setsid nohup bash tools/download_peod.sh > ~/work/shared_data/PEOD/download.out 2>&1 &
set -u

ROOT=${1:-$HOME/work/shared_data}
DIR=$ROOT/PEOD
ARC=$DIR/_archives
LOG=$DIR/download.log
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
URL='https://drive.usercontent.google.com/download?export=download&confirm=t&id='
mkdir -p "$ARC"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }
log start

# name, then file ids to try in order, small to large. First id: a copy of the same files in the Drive folder
# (a copy of the files; folder and file ids kept out of the public repo) (given by the user 2026-11-25; sizes
# and sampled bytes match the original). Second id: the authors' original folder (above). Each copy has its own
# download quota, and both serve identical bytes, so a partial file can resume from either.
FILES=(
  "annotations.zip 15J7fLoA9a0mHrWc1dWsAXHT65Fl9mCPl"
  "timestamp.zip 1CE-g67-hBwueWKX3QgKWxDhHX4BL_tQk"
  "sequence_024-5.zip 1nH8HzC6Jppqbk4Uhub7x45UlQ25mYh88"
  "rgb.tar.gz 1x1-VnGSbfBaT64eGzMTsYItLTixbSXMG"
  "Event.tar.gz.part-000 1ytm2Eyu5DGDxlhXsa5zUagLMQkDN-6Vu"
  "Event.tar.gz.part-001 1fR20FO6oLqakF1PDp9-uVd5M01PBx4vP"
  "Event.tar.gz.part-002 1KlhyrUC1FzTHSVZ0pzmXNndG-eO3yIzl"
)

# Each Drive file has its own download quota, so a file whose quota is exceeded is set aside and the others are
# tried; set-aside files are retried in the next pass (immediately if the pass finished a file, else after 1 h).
pending=("${FILES[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  left=() progress=0
  for entry in "${pending[@]}"; do
    read -r name ids <<< "$entry"
    out=$ARC/$name
    if [ -f "$out.done" ]; then log "skip (done): $name"; continue; fi
    rc=1
    for id in $ids; do               # next copy only when this one is over quota
      log "download: $name (id $id)"
      fails=0
      while [ $fails -lt 5 ]; do
        "$PYTHON" "$HERE/pdownload.py" "$URL$id" "$out" --workers 4
        rc=$?
        if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then break; fi
        fails=$((fails + 1)); log "  pdownload failed ($rc), retry $fails"; sleep 60
      done
      [ $rc -eq 3 ] || break
    done
    if [ $rc -eq 3 ]; then          # Drive download quota of this shared file (progress kept)
      log "  Google Drive quota exceeded for $name; set aside, trying the other files"
      left+=("$entry"); continue
    fi
    [ $rc -eq 0 ] || { log "FAILED download: $name"; exit 1; }
    # Drive returns a small HTML page instead of the file when its quota is exceeded
    if head -c 15 "$out" | grep -qi '<!doctype\|<html'; then log "FAILED: $name is an HTML page (Drive quota?)"; exit 1; fi
    if [ ! -s "$out" ]; then log "FAILED: $name is empty"; exit 1; fi
    # a sparse file (truncated to full size but never written) passes -s; require the bytes to be on disk
    if [ $(( $(stat -c %b "$out") * 512 )) -lt $(( $(stat -c %s "$out") * 99 / 100 )) ]; then
      log "FAILED: $name is mostly unwritten (sparse)"; exit 1
    fi
    touch "$out.done"
    log "  size $(du -h "$out" | cut -f1)"
    progress=1
  done
  pending=("${left[@]+"${left[@]}"}")
  if [ ${#pending[@]} -gt 0 ] && [ $progress -eq 0 ]; then
    log "  all remaining files are over quota; retry in 1 h"; sleep 3600
  fi
done

# extract (archives kept)
x() {
  local n=$1; shift
  [ -f "$DIR/.x_$n" ] && return 0
  log "extract $n"
  "$@" || { log "FAILED extract $n"; exit 1; }
  touch "$DIR/.x_$n"
}
x annotations unzip -q -n "$ARC/annotations.zip" -d "$DIR"
x timestamp   unzip -q -n "$ARC/timestamp.zip" -d "$DIR"
x seq024_5    unzip -q -n "$ARC/sequence_024-5.zip" -d "$DIR/_sequence_024-5"
mkdir -p "$DIR/rgb" "$DIR/event"     # rgb.tar.gz paths start with ./train, ./test (no rgb/ prefix)
x rgb         tar -xzf "$ARC/rgb.tar.gz" -C "$DIR/rgb"
x event       sh -c "cat '$ARC'/Event.tar.gz.part-00? | tar -xz -C '$DIR/event'"
log "top level: $(ls "$DIR" | tr '\n' ' ')"
log "png: $(find "$DIR" -name '*.png' | wc -l), dat: $(find "$DIR" -name '*.dat' | wc -l), free: $(df -h "$ROOT" | awk 'NR==2 {print $4}')"
log 'ALL DONE'

# then convert to the training layout (tools/preprocess_peod.py); CONVERT=0 skips this
if [ "${CONVERT:-1}" = 1 ]; then
  log "preprocess -> $ROOT/PEOD_train (log: $DIR/preprocess.log)"
  ( source "$HOME/work/hsf_env.sh" && cd "$HERE/.." && \
    python -u tools/preprocess_peod.py --root "$DIR" --out "$ROOT/PEOD_train" --workers 12 ) >> "$DIR/preprocess.log" 2>&1
  log "preprocess exit $? ($(grep -c FAILED "$DIR/preprocess.log") FAILED lines)"
fi
