#!/bin/bash
set -e
source /workspace/slippi-ai/.venv/bin/activate
cd /workspace/slippi-ai

# Extracted ExiAI AppImage (no FUSE). libmelee wants a path containing
# "netplay" and a binary named Slippi_Online-x86_64.AppImage.
# Do NOT use AppRun: RunPod's huge LD_LIBRARY_PATH triggers E2BIG.
ROOT=/workspace/dolphin/squashfs-root
NETPLAY=/workspace/dolphin/netplay
WRAPPER="$NETPLAY/Slippi_Online-x86_64.AppImage"
mkdir -p "$NETPLAY"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
ROOT=/workspace/dolphin/squashfs-root
DOLPHIN_LD="$ROOT/usr/lib:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib:$ROOT/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
# Clean env for dolphin only — RunPod CUDA paths make argv/env too large for AppRun.
exec env -i \
  HOME="${HOME:-/root}" \
  USER="${USER:-root}" \
  PATH="/usr/bin:/bin:/usr/local/bin" \
  LD_LIBRARY_PATH="$DOLPHIN_LD" \
  "$ROOT/usr/bin/dolphin-emu" "$@"
EOF
chmod +x "$WRAPPER"
export DOLPHIN_PATH="$NETPLAY"

# Smoke-test dolphin before launching training.
if ! "$WRAPPER" --version >/tmp/dolphin_version.txt 2>&1; then
  echo "Dolphin failed to start. Output:"
  cat /tmp/dolphin_version.txt || true
  echo "Missing libs (if any):"
  env -i LD_LIBRARY_PATH="$ROOT/usr/lib:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib:$ROOT/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu" \
    ldd "$ROOT/usr/bin/dolphin-emu" 2>/dev/null | grep "not found" || true
  exit 1
fi
echo "Dolphin OK: $(tr '\n' ' ' </tmp/dolphin_version.txt)"

export ISO_PATH="/workspace/iso/Super Smash Bros. Melee (USA) (v1.02).iso"
export TEACHER=/workspace/models/medium-v2
export EXPT_DIR=/workspace/experiments/fox-stay-on-stage
mkdir -p "$EXPT_DIR"
: > "$EXPT_DIR/train.log"
nohup python slippi_ai/rl/run.py \
  --config.runtime.tag=fox-stay-on-stage \
  --config.runtime.expt_dir="$EXPT_DIR" \
  --config.runtime.max_runtime=7200 \
  --config.runtime.log_interval=60 \
  --config.runtime.save_interval=300 \
  --config.dolphin.path="$DOLPHIN_PATH" \
  --config.dolphin.iso="$ISO_PATH" \
  --config.dolphin.headless=True \
  --config.dolphin.emulation_speed=0 \
  --config.teacher="$TEACHER" \
  --config.opponent.type=self \
  --config.opponent.train=True \
  --config.actor.num_envs=32 \
  --config.actor.rollout_length=120 \
  --config.actor.inner_batch_size=8 \
  --config.actor.async_envs=True \
  --config.actor.gpu_inference=True \
  --config.agent.char=FOX \
  --wandb.mode=disabled \
  > "$EXPT_DIR/train.log" 2>&1 &
echo "PID=$!"
