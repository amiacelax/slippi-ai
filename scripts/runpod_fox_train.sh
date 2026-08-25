#!/bin/bash
set -e
source /workspace/slippi-ai/.venv/bin/activate
cd /workspace/slippi-ai

# Match GitHub CI: mainline ExiAI AppImage + libfuse2 (file path, not extract).
DOLPHIN_DIR=/workspace/dolphin
DOLPHIN_APPIMAGE="$DOLPHIN_DIR/Slippi_Netplay_Mainline_ExiAI-x86_64.AppImage"
DOLPHIN_URL="https://github.com/vladfi1/dolphin/releases/download/4.0.0-mainline-beta.14-ExiAI/Slippi_Netplay_Mainline_ExiAI-x86_64.AppImage"

echo "Installing dolphin runtime deps (libfuse2, libegl1)..."
apt-get update -qq
apt-get install -y -qq libfuse2 libegl1 >/dev/null

mkdir -p "$DOLPHIN_DIR"
if [ ! -f "$DOLPHIN_APPIMAGE" ]; then
  echo "Downloading mainline ExiAI AppImage..."
  curl -L --fail --retry 3 -o "$DOLPHIN_APPIMAGE" "$DOLPHIN_URL"
fi
chmod +x "$DOLPHIN_APPIMAGE"

# Works without FUSE on many containers; harmless if FUSE is available.
export APPIMAGE_EXTRACT_AND_RUN=1
export DOLPHIN_PATH="$DOLPHIN_APPIMAGE"

echo "Smoke-testing dolphin (15s timeout)..."
set +e
timeout 15 "$DOLPHIN_PATH" --version >/tmp/dolphin_version.txt 2>&1
rc=$?
set -e
if [ "$rc" -eq 124 ]; then
  echo "WARNING: dolphin --version hung; training will use timeout fallback."
  cat /tmp/dolphin_version.txt || true
elif [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ] && [ "$rc" -ne 255 ]; then
  echo "Dolphin failed to start (exit $rc). Output:"
  cat /tmp/dolphin_version.txt || true
  exit 1
else
  echo "Dolphin OK (exit $rc): $(tr '\n' ' ' </tmp/dolphin_version.txt)"
fi

export ISO_PATH="/workspace/iso/Super Smash Bros. Melee (USA) (v1.02).iso"
export TEACHER=/workspace/models/medium-v2
export EXPT_DIR=/workspace/experiments/fox-stay-on-stage
mkdir -p "$EXPT_DIR"
: > "$EXPT_DIR/train.log"

# TF RL loop uses max_step (max_runtime is ignored here).
# ~7s/step at 8 envs → 1200 steps ≈ 2+ hours.
echo "Starting training..."
export PYTHONUNBUFFERED=1
nohup python slippi_ai/rl/run.py \
  --config.runtime.tag=fox-stay-on-stage \
  --config.runtime.expt_dir="$EXPT_DIR" \
  --config.runtime.max_step=1200 \
  --config.runtime.log_interval=60 \
  --config.runtime.save_interval=300 \
  --config.dolphin.path="$DOLPHIN_PATH" \
  --config.dolphin.iso="$ISO_PATH" \
  --config.dolphin.headless=True \
  --config.dolphin.emulation_speed=0 \
  --config.teacher="$TEACHER" \
  --config.opponent.type=self \
  --config.opponent.train=True \
  --config.actor.num_envs=8 \
  --config.actor.rollout_length=120 \
  --config.actor.inner_batch_size=4 \
  --config.actor.async_envs=True \
  --config.actor.gpu_inference=True \
  --config.agent.char=FOX \
  --wandb.mode=disabled \
  > "$EXPT_DIR/train.log" 2>&1 &
echo "PID=$!"
echo "Log: $EXPT_DIR/train.log"
