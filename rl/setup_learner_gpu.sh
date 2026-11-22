#!/usr/bin/env bash
# One-shot setup for the HIL-SERL *learner* on a remote GPU machine.
# Usage (on the GPU box, from the repo root):
#   bash rl/setup_learner_gpu.sh            # install + checks
#   bash rl/setup_learner_gpu.sh run        # install + start learner
#
# The learner is the gRPC *server*; the actor (your laptop) connects to it.
# Default is to bind 127.0.0.1 and reach it through an SSH tunnel from the laptop
# (see bottom of this file). Set LEARNER_BIND=0.0.0.0 to expose the port directly
# (only on a private network / Tailscale / firewalled host: the channel is insecure gRPC).
set -euo pipefail

CONFIG="${CONFIG:-rl/train_config_sim.json}"
PORT="${PORT:-50051}"
LEARNER_BIND="${LEARNER_BIND:-127.0.0.1}"

# 1. Dependencies
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra hilserl --extra training  # `training` provides wandb

# 2. Sanity: GPU visible
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

# 3. Auth (dataset szymmon/cube_pickup comes from the Hub; wandb is enabled in the config)
[ -n "${HF_TOKEN:-}" ] && uv run hf auth login --token "$HF_TOKEN" --add-to-git-credential || \
  uv run hf auth whoami >/dev/null 2>&1 || { echo "Run: uv run hf auth login   (or export HF_TOKEN)"; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || uv run wandb status 2>/dev/null | grep -q api_key || \
  echo "WARN: wandb not logged in (export WANDB_API_KEY or run 'uv run wandb login'), or set wandb.enable=false"

# 4. Learner-side config: same file as the laptop, but with the bind address overridden.
LEARNER_CFG="rl/train_config.learner.json"
uv run python - "$CONFIG" "$LEARNER_CFG" "$LEARNER_BIND" "$PORT" <<'EOF'
import json, sys
src, dst, host, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.load(open(src))
cfg["policy"]["actor_learner_config"]["learner_host"] = host
cfg["policy"]["actor_learner_config"]["learner_port"] = port
cfg["policy"]["device"] = cfg["policy"]["storage_device"] = "cuda"
json.dump(cfg, open(dst, "w"), indent=4)
print(f"wrote {dst} (learner binds {host}:{port})")
EOF

if [ "${1:-}" = "run" ]; then
  exec uv run python -m lerobot.rl.learner --config_path "$LEARNER_CFG"
fi
echo "Ready. Start with: uv run python -m lerobot.rl.learner --config_path $LEARNER_CFG"

# ---- On the LAPTOP (actor = local MuJoCo sim + keyboard interventions) --------
#   ssh -N -L 50051:127.0.0.1:50051 user@GPU_HOST        # leave running
#   uv run python -m lerobot.rl.actor --config_path rl/train_config_sim.json \
#       --policy.device=mps --policy.storage_device=cpu
# (on macOS the MuJoCo viewer may need `mjpython -m lerobot.rl.actor ...` instead)
# Direct (no tunnel): LEARNER_BIND=0.0.0.0 here, and add --policy.actor_learner_config.learner_host=GPU_IP on the laptop.
