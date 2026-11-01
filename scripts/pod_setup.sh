#!/bin/bash
# Installs the pinned training stack on a rented GPU machine. Runs on the pod.
set -euo pipefail

PRIME_RL_COMMIT=a566e5d8e2a903f44d901e114d5d510888b93303
WORK=${WORK:-/root/work}
export PATH=$HOME/.local/bin:$PATH
export GIT_TERMINAL_PROMPT=0

mkdir -p "$WORK"
cd "$WORK"
curl -LsSf https://astral.sh/uv/install.sh | sh        # the image may carry an uv too old for the lock file
command -v git > /dev/null || { apt-get update -q && apt-get install -yq git; }

if [ ! -d prime-rl/.git ]; then
    git init -q prime-rl
    git -C prime-rl remote add origin https://github.com/PrimeIntellect-ai/prime-rl.git
    git -C prime-rl fetch -q --depth 1 origin "$PRIME_RL_COMMIT"
    git -C prime-rl checkout -q FETCH_HEAD
fi
cd prime-rl
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --depth 1 -q -- \
    deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync -q --all-extras
uv pip install -q scipy
uv pip install -q --no-deps -e "$WORK/geotech-rl" -e "$WORK/geotech-rl/envs/geotech_poles"

.venv/bin/python - <<'PY'
import flash_attn, geotech_poles, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1), "GiB")
PY
echo POD_READY
