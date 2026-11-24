#!/usr/bin/env bash
# A quick check of an installation on a CPU: every step of reproduce.sh, with a few training steps per model (the numbers mean
# nothing). Scores the validation set only; evaluates the published scores if the dataset folder has baselines/.
#   scripts/smoke_test.sh <dataset folder> [work folder]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?usage: scripts/smoke_test.sh <dataset folder> [work folder]}
WORK=${2:-smoke}
A="$WORK/arrays"; R="$WORK/runs"; S="$WORK/scores/validation"
[ -f "$A/meta.json" ] || python models/prepare.py --data "$DATA" --out "$A"
FAST="--steps 4 --bs 8 --device cpu --ckpt_every 0"
python models/train.py --arrays "$A" --recipe teacher-5m --out "$R/teacher-5m-seed0" $FAST --eval_every 0
python models/train.py --arrays "$A" --recipe student-2.6m-distilled --teacher "$R/teacher-5m-seed0/model.safetensors" --out "$R/student-2.6m-distilled-seed0" $FAST --eval_every 0
python models/train.py --arrays "$A" --recipe student-0.4m-distilled --teacher "$R/teacher-5m-seed0/model.safetensors" --out "$R/student-0.4m-distilled-seed0" $FAST --eval_every 2
python models/train.py --arrays "$A" --recipe student-0.4m-labels-only --teacher "$R/teacher-5m-seed0/model.safetensors" --out "$R/student-0.4m-labels-only-seed0" $FAST --eval_every 0
mkdir -p "$S"
for run in "$R"/*-seed0/; do
  python models/score.py --arrays "$A" --model "$run/model.safetensors" --split validation --device cpu --out "$S/$(basename "$run").parquet"
done
if [ -f "$DATA/baselines/weights/xgboost-geo.json" ]; then
  python baselines/xgboost/predict.py --arrays "$A" --model "$DATA/baselines/weights/xgboost-geo.json" --split validation --out "$S/xgboost-geo.parquet"
fi
if [ -d "$DATA/baselines/scores/test" ]; then
  python evaluate.py --data "$DATA" --scores "$DATA"/baselines/scores/test/student-0.4m-distilled-seed{0,1,2}.parquet \
                     --dev-scores "$DATA"/baselines/scores/validation/student-0.4m-distilled-seed{0,1,2}.parquet
fi
echo "smoke test passed"
