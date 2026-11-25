#!/usr/bin/bash
# Paper schedule (HSF Sec. IV-B: 40 epochs, SGD lr 0.005, batch 8, warmup 3 = train_dual.py defaults)
# for the four ablation rows of HSF Table II, checkpoints selected on the validation split.
# One worker per GPU in GPUS; each worker runs its share of the fusions one after another, so at most
# len(GPUS) GPUs are busy (default 2: the shared lab server must not have all 4 GPUs taken).
# With GPUS="0 3": GPU 0 runs add then af, GPU 3 runs hsf then tfs (as in the 2026-11-12 seed-0 runs).
# INIT selects the step-3 single-stream initialisation: "" = runs/step3/{rgb,event}_yolov8s (all 47 sequences,
# selected on test; the *_paper_s0 runs), "_val" = runs/step3/{rgb,event}_yolov8s_val (train_core, whole-drive
# val; since 2026-11-14). The run name gets the suffix "_init<INIT>" when INIT is set.
# Usage (from the repo root, inside the training env):
#   SEED=0 GPUS="0 3" INIT=_val setsid nohup bash tools/run_paper_schedule.sh > runs/paper_schedule.log 2>&1 < /dev/null &
# setsid: the runs must survive the closing of the terminal that started them.
cd "$(dirname "$0")/.."
SEED=${SEED:-0}
INIT=${INIT:-}
read -r -a G <<< "${GPUS:-0 1}"
FUSIONS=(add hsf af tfs)
S3=runs/step3
INIT_ARGS=(--rgb-init "$S3/rgb_yolov8s$INIT/weights/best.pt" --evt-init "$S3/event_yolov8s$INIT/weights/best.pt")
for f in "${INIT_ARGS[@]:1:1}" "${INIT_ARGS[@]:3:1}"; do
    [ -f "$f" ] || { echo "missing init checkpoint $f"; exit 1; }
done

worker() {      # worker <slot>: fusions slot, slot+N, ... on GPU G[slot]
    local slot=$1 gpu=${G[$1]} i f name
    for ((i = slot; i < ${#FUSIONS[@]}; i += ${#G[@]})); do
        f=${FUSIONS[$i]}
        name=dual_${f}_paper_s${SEED}${INIT:+_init$INIT}
        echo "[$(date '+%F %T')] start $name on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion "$f" --seed "$SEED" --name "$name" \
            "${INIT_ARGS[@]}" > "runs/${name}.out" 2>&1
        echo "[$(date '+%F %T')] end $name (exit $?)"
    done
}

for s in "${!G[@]}"; do
    worker "$s" &
    sleep 60            # stagger start-up (dataset indexing reads every meta.json)
done
wait
echo "[$(date '+%F %T')] all done"
