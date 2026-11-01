#!/bin/bash
# Trains on one GPU: inference starts first and the trainer shares the card. Runs where the GPU is.
#   CONFIG=configs/pilot.toml OUT=/root/work/t4 LIMIT_MIN=30 bash train.sh [extra rl arguments]
# Two ranks of the same card cannot form an NCCL group, so weights go through the file system.
set -uo pipefail

MODEL=${MODEL:-/root/work/models/Qwen3-4B-Instruct-2507-sft}
LIMIT_MIN=${LIMIT_MIN:-30}
MAX_LEN=${MAX_LEN:-32768}
UTIL=${UTIL:-0.45}
RANK=${RANK:-16}
PARSER=${PARSER:-hermes}
OUT=${OUT:-outputs/train}
PRIME=${PRIME:-/root/work/prime-rl}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$REPO/configs/pilot.toml}
export PATH=$HOME/.local/bin:$PATH

cd "$PRIME"
mkdir -p "$OUT"
started=$(date +%s)
uv run --no-sync python "$REPO/scripts/preflight.py" || exit 1
uv run --no-sync inference --vllm.model "$MODEL" --vllm.tool-call-parser "$PARSER" --vllm.enable-lora --vllm.max-lora-rank "$RANK" \
    --vllm.gpu-memory-utilization "$UTIL" --vllm.max-model-len "$MAX_LEN" --enable-return-sampling-mask True > "$OUT/inference.log" 2>&1 &
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
timeout -k 20 "$left" uv run --no-sync rl @ "$CONFIG" --no-inference --deployment.num-infer-gpus 0 \
    --orchestrator.model.client.admin-base-url http://localhost:8100 --weight-broadcast.type filesystem \
    --no-monitors.prime --no-monitors.wandb --output-dir "$OUT" "$@" > "$OUT/rl.log" 2>&1
echo "rl: exit $? after $(($(date +%s) - started)) s"
sort -n "$OUT/gpu.log" | tail -1 | sed 's/^/peak gpu: /'
echo TRAIN_DONE
