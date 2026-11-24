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

# name id, small to large
FILES=(
  "annotations.zip 15J7fLoA9a0mHrWc1dWsAXHT65Fl9mCPl"
  "timestamp.zip 1CE-g67-hBwueWKX3QgKWxDhHX4BL_tQk"
  "sequence_024-5.zip 1nH8HzC6Jppqbk4Uhub7x45UlQ25mYh88"
  "rgb.tar.gz 1x1-VnGSbfBaT64eGzMTsYItLTixbSXMG"
  "Event.tar.gz.part-000 1ytm2Eyu5DGDxlhXsa5zUagLMQkDN-6Vu"
  "Event.tar.gz.part-001 1fR20FO6oLqakF1PDp9-uVd5M01PBx4vP"
  "Event.tar.gz.part-002 1KlhyrUC1FzTHSVZ0pzmXNndG-eO3yIzl"
)

for entry in "${FILES[@]}"; do
  read -r name id <<< "$entry"
  out=$ARC/$name
  if [ -f "$out.done" ]; then log "skip (done): $name"; continue; fi
  log "download: $name"
  ok=0
  for try in 1 2 3 4 5; do
    if "$PYTHON" "$HERE/pdownload.py" "$URL$id" "$out" --workers 4; then ok=1; break; fi
    log "  pdownload failed, retry $try"; sleep 60
  done
  if [ $ok -ne 1 ]; then
    log "  falling back to curl -C -"
    curl -sSL -C - --retry 20 -o "$out" "$URL$id" && ok=1
  fi
  [ $ok -eq 1 ] || { log "FAILED download: $name"; exit 1; }
  # Drive returns a small HTML page instead of the file when its quota is exceeded
  if head -c 15 "$out" | grep -qi '<!doctype\|<html'; then log "FAILED: $name is an HTML page (Drive quota?)"; exit 1; fi
  if [ ! -s "$out" ]; then log "FAILED: $name is empty"; exit 1; fi
  touch "$out.done"
  log "  size $(du -h "$out" | cut -f1)"
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
x rgb         tar -xzf "$ARC/rgb.tar.gz" -C "$DIR"
x event       sh -c "cat '$ARC'/Event.tar.gz.part-00? | tar -xz -C '$DIR'"
log "top level: $(ls "$DIR" | tr '\n' ' ')"
log "png: $(find "$DIR" -name '*.png' | wc -l), dat: $(find "$DIR" -name '*.dat' | wc -l), free: $(df -h "$ROOT" | awk 'NR==2 {print $4}')"
log 'ALL DONE'
