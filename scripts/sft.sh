#!/bin/bash
# Supervised warm start on one GPU: demonstrations, then inference for the evals, then the trainer.
#   OUT=/root/work/t5 LIMIT_MIN=35 DEMOS=600 bash sft.sh [extra sft arguments]
set -uo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LIMIT_MIN=${LIMIT_MIN:-35}
DEMOS=${DEMOS:-600}
LEVELS=${LEVELS:-0,1}
MAX_LEN=${MAX_LEN:-32768}
UTIL=${UTIL:-0.50}
RANK=${RANK:-16}
PARSER=${PARSER:-hermes}
OUT=${OUT:-outputs/sft}
DATA=${DATA:-/root/work/demos}
PRIME=${PRIME:-/root/work/prime-rl}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$REPO/configs/sft.toml}
export PATH=$HOME/.local/bin:$PATH

cd "$PRIME"
mkdir -p "$OUT"
started=$(date +%s)
uv run --no-sync python "$REPO/scripts/preflight.py" || exit 1
[ -f "$DATA/train.jsonl" ] || uv run --no-sync python "$REPO/scripts/demonstrations.py" "$DATA" "$DEMOS" "$LEVELS" || exit 1
echo "demonstrations ready after $(($(date +%s) - started)) s"
uv run --no-sync inference --vllm.model "$MODEL" --vllm.tool-call-parser "$PARSER" --vllm.enable-lora --vllm.max-lora-rank "$RANK" \
    --vllm.gpu-memory-utilization "$UTIL" --vllm.max-model-len "$MAX_LEN" > "$OUT/inference.log" 2>&1 &
server=$!
(while sleep 15; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; done) > "$OUT/gpu.log" &
monitor=$!
trap 'pkill -f prime_rl; pkill -f vllm; kill $server $monitor 2> /dev/null' EXIT

for _ in $(seq 1 180); do
    kill -0 $server 2> /dev/null || { echo "inference died"; tail -5 "$OUT/inference.log"; exit 1; }
    curl -s -m 2 localhost:8000/v1/models | grep -q "$(basename "$MODEL")" && break
    sleep 5
done
echo "inference ready after $(($(date +%s) - started)) s"

left=$((LIMIT_MIN * 60 - ($(date +%s) - started)))
timeout -k 20 "$left" uv run --no-sync sft @ "$CONFIG" --deployment.num-infer-gpus 0 \
    --weight-broadcast.type filesystem --no-monitors.prime --no-monitors.wandb --output-dir "$OUT" "$@" > "$OUT/sft.log" 2>&1
echo "sft: exit $? after $(($(date +%s) - started)) s"
sort -n "$OUT/gpu.log" | tail -1 | sed 's/^/peak gpu: /'
echo TRAIN_DONE
