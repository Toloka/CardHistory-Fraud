#!/usr/bin/env bash
# Retrain and score every released baseline, then print the table (on a GPU; a CPU works but takes days).
#   scripts/reproduce.sh <dataset folder> [work folder]
# Environment: SEEDS (default "0 1 2"); TEACHER (default: the retrained Teacher 5M seed 0, as in our run; set it to
# <dataset folder>/baselines/weights/teacher-5m-seed0.safetensors to distill from the released teacher); XGB_DEVICE (default cuda).
# Finished steps are skipped, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?usage: scripts/reproduce.sh <dataset folder> [work folder]}
WORK=${2:-work}
SEEDS=${SEEDS:-"0 1 2"}
A="$WORK/arrays"; R="$WORK/runs"; S="$WORK/scores"
[ -f "$A/meta.json" ] || python models/prepare.py --data "$DATA" --out "$A"

train() {   # recipe seed [teacher]
  local out="$R/$1-seed$2"
  [ -f "$out/model.safetensors" ] && return 0
  python models/train.py --arrays "$A" --recipe "$1" --seed "$2" --out "$out" ${3:+--teacher "$3"}
}
for s in $SEEDS; do train teacher-5m "$s"; done
TEACHER=${TEACHER:-"$R/teacher-5m-seed0/model.safetensors"}
for m in student-2.6m-distilled student-0.4m-distilled student-0.4m-labels-only; do
  for s in $SEEDS; do train "$m" "$s" "$TEACHER"; done
done
[ -f "$R/xgboost-geo/model.json" ] || python baselines/xgboost/train.py --arrays "$A" --out "$R/xgboost-geo/model.json" --device "${XGB_DEVICE:-cuda}"

for split in test validation; do
  mkdir -p "$S/$split"
  for run in "$R"/*-seed*/; do
    n=$(basename "$run")
    [ -f "$S/$split/$n.parquet" ] || python models/score.py --arrays "$A" --model "$run/model.safetensors" --split "$split" --out "$S/$split/$n.parquet"
  done
  [ -f "$S/$split/xgboost-geo.parquet" ] || python baselines/xgboost/predict.py --arrays "$A" --model "$R/xgboost-geo/model.json" --split "$split" --out "$S/$split/xgboost-geo.parquet"
done

for m in teacher-5m student-2.6m-distilled student-0.4m-distilled student-0.4m-labels-only; do
  t=(); v=()
  for s in $SEEDS; do t+=("$S/test/$m-seed$s.parquet"); v+=("$S/validation/$m-seed$s.parquet"); done
  echo "== $m"
  python evaluate.py --data "$DATA" --scores "${t[@]}" --dev-scores "${v[@]}"
done
echo "== xgboost-geo"
python evaluate.py --data "$DATA" --scores "$S/test/xgboost-geo.parquet" --dev-scores "$S/validation/xgboost-geo.parquet"
