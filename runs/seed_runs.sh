#!/bin/bash
# Extra seeds for the main fusions (2026-11-16): add, relp2_add, sw2_add, cafr with seeds 1 and 2
# (seed 0 already exists). Same short schedule as runs/rel_iter.sh: 5 epochs, lr 1e-3, warmup 1,
# clean initialisation, checkpoints selected on the val split. Never more than 2 GPUs (0 and 3).
#   1. training: one queue per GPU, balanced by run time (cafr ~2.8 h, the others ~1.1 h)
#   2. per-lighting evaluation of all 8 checkpoints (one process, so runs/eval_conditions.json is not raced)
#   3. RGB corruption robustness, one process per GPU, each writing its own json/md
# Usage: setsid nohup bash runs/seed_runs.sh > runs/seed_runs.log 2>&1 < /dev/null &
# Writes runs/seed_runs_train.done, runs/seed_runs.done.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
S3=runs/step3
COMMON="--epochs 5 --lr 1e-3 --warmup 1 --rgb-init $S3/rgb_yolov8s_val/weights/best.pt --evt-init $S3/event_yolov8s_val/weights/best.pt"
log() { echo "[$(date '+%F %T')] $*"; }
dir_of() { case $1 in add) echo step4;; rel*) echo step11_rel;; sw*) echo step12_sw;; cafr*) echo step13_cafr;; esac; }
name_of() { echo "dual_$1_s5_seed$2"; }     # name_of <fusion> <seed>
train() {   # train <gpu> <fusion> <seed>
    local gpu=$1 f=$2 seed=$3 name; name=$(name_of "$2" "$3")
    if [ -f "runs/$(dir_of "$f")/$name/test_metrics.json" ]; then
        log "skip $name (already trained)"; return
    fi
    log "start $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion "$f" --name "$name" --seed "$seed" $COMMON \
        > "runs/${name}.out" 2>&1
    log "end $name (exit $?)"
}
queue() {   # queue <gpu> <fusion:seed>...
    local gpu=$1; shift
    for spec in "$@"; do train "$gpu" "${spec%%:*}" "${spec##*:}"; done
}
spec_of() { local n; n=$(name_of "$1" "$2"); echo "${1}_seed$2=$(dir_of "$1")/$n/best.pt:$1"; }   # eval key = <fusion>_seed<n>

rm -f runs/seed_runs_train.done runs/seed_runs.done
log "training"
queue 0 cafr:1 add:1 add:2 relp2_add:1 &
sleep 45
queue 3 cafr:2 sw2_add:1 sw2_add:2 relp2_add:2 &
wait
touch runs/seed_runs_train.done

G0=(); G3=()
for s in 1 2; do G0+=("$(spec_of add $s)"); G3+=("$(spec_of sw2_add $s)"); done
G0+=("$(spec_of relp2_add 1)" "$(spec_of cafr 1)")
G3+=("$(spec_of relp2_add 2)" "$(spec_of cafr 2)")
log "per-lighting evaluation (GPU 0) + corruption evaluation (GPU 3): ${G0[*]} ${G3[*]}"
CUDA_VISIBLE_DEVICES=0 python tools/eval_conditions.py --models --extra "${G0[@]}" "${G3[@]}" \
    > runs/seed_runs_eval_conditions.log 2>&1 &
P_COND=$!
CUDA_VISIBLE_DEVICES=3 python tools/eval_corruption.py --models --extra "${G3[@]}" --out eval_corruption_seeds_g3 \
    > runs/seed_runs_corruption_g3.log 2>&1 &
wait $P_COND
log "corruption evaluation (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python tools/eval_corruption.py --models --extra "${G0[@]}" --out eval_corruption_seeds_g0 \
    > runs/seed_runs_corruption_g0.log 2>&1
wait
log done
touch runs/seed_runs.done
