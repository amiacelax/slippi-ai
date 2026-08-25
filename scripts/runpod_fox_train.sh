#!/bin/bash
set -e
source /workspace/slippi-ai/.venv/bin/activate
cd /workspace/slippi-ai
export DOLPHIN_PATH=/workspace/dolphin/squashfs-root/usr/bin
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
