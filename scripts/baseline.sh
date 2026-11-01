#!/bin/bash
# Measures an open model on each level, on held-out instances, before any training. Runs where the GPU is.
#   MODEL=... LEVELS=0,1 EPISODES=20 GROUP=3 LIMIT_MIN=35 bash baseline.sh
# A canary of CANARY tasks runs first: the environment gates and the throughput are known in minutes, and the
# full run shrinks to what fits in LIMIT_MIN. Nothing here can run past the limit.
set -uo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
LEVELS=${LEVELS:-0 1}
EPISODES=${EPISODES:-20}
GROUP=${GROUP:-3}
CANARY=${CANARY:-4}
LIMIT_MIN=${LIMIT_MIN:-35}
MAX_LEN=${MAX_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-2048}
UTIL=${UTIL:-0.90}
OUT=${OUT:-outputs/baseline}
SPLIT=${SPLIT:-heldout}
INLINE=${INLINE:-false}
PARSER=${PARSER:-hermes}
EXTRA=${EXTRA:-}
PRIME=${PRIME:-/root/work/prime-rl}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/.local/bin:$PATH

cd "$PRIME"
mkdir -p "$OUT"
started=$(date +%s)
left() { echo $((LIMIT_MIN * 60 - ($(date +%s) - started))); }

evaluate() {  # level, tasks, output directory
    local config="$3.toml"
    cat > "$config" <<TOML
model = "$MODEL"
group_size = $GROUP

[select]
limit = $2

[client]
base_url = "http://localhost:8000/v1"

[sampling]
max_completion_tokens = $MAX_TOKENS

[[source]]
name = "geotech-poles"
env.taskset.id = "geotech-poles"
env.taskset.levels = [$1]
env.taskset.split = "$SPLIT"
env.taskset.num_seeds = $2
env.taskset.inline = $INLINE
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
TOML
    local begun=$(date +%s)
    timeout "$(left)" uv run --no-sync eval @ "$config" --no-monitors.prime --no-monitors.wandb --output-dir "$3" > "$3.log" 2>&1
    local status=$?
    echo "$3: exit $status, $2 x $GROUP episodes in $(($(date +%s) - begun)) s"
    uv run --no-sync python "$REPO/scripts/summarise_eval.py" "$3" --strict || { echo "ABORTED at $3"; exit 1; }
}

uv run --no-sync python "$REPO/scripts/preflight.py" || exit 1
uv run --no-sync inference --vllm.model "$MODEL" --vllm.tool-call-parser "$PARSER" $EXTRA \
    --vllm.gpu-memory-utilization "$UTIL" --vllm.max-model-len "$MAX_LEN" > "$OUT/inference.log" 2>&1 &
server=$!
trap 'pkill -f vllm; kill $server 2> /dev/null' EXIT

for _ in $(seq 1 180); do
    kill -0 $server 2> /dev/null || { echo "inference died"; tail -5 "$OUT/inference.log"; exit 1; }
    curl -s -m 2 localhost:8000/v1/models | grep -q "$(basename "$MODEL")" && break
    sleep 5
done
echo "inference ready after $(($(date +%s) - started)) s"

set -- ${LEVELS//,/ }
begun=$(date +%s)
evaluate "$1" "$CANARY" "$OUT/canary_n$1"
per_task=$((($(date +%s) - begun) / CANARY + 1))
fits=$(($(left) / per_task / $#))
tasks=$((fits < EPISODES ? fits : EPISODES))
echo "canary: at most $per_task s per task, $(left) s left, running $tasks tasks x $GROUP per level"
[ "$tasks" -ge 3 ] || { echo "ABORTED: too slow for the limit"; exit 1; }

for level in "$@"; do
    evaluate "$level" "$tasks" "$OUT/n$level"
done

uv run --no-sync python "$REPO/scripts/summarise_eval.py" "$OUT" | tee "$OUT/summary.json"
echo "BASELINE_DONE total $(($(date +%s) - started)) s"
