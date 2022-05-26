#!/usr/bin/env bash

ENV="Carla-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/TINY_CNN/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/TINY_CNN"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

# -0.1 every time step, more random spawn in route, float32 training
# perfect sample action at start

python main.py \
	--mode train --gpu 0 --sampler-gpu 1 \
	--env "$ENV" \
	--n-frames 5 \
	--grayscale \
	--n-past-actions 10 \
	--image-size 80 160 \
	--encoder-arch TINY_CNN \
	--activation LeakyReLU \
	--state-dim 256 \
	--hidden-dims 512 256 \
	--max-episode-steps 5000 \
	--n-epochs 1000 --n-updates 256 --batch-size 64 \
	--n-samplers 1 \
	--buffer-capacity 30000 \
	--update-sample-ratio 5.0 \
	--critic-lr 3E-4 --actor-lr 3E-4 \
	--alpha-lr 3E-4 --initial-alpha 1.0 \
	--adaptive-entropy --target-entropy -3 \
	--n-bootstrap-step 5 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--log-episode-video \
	--dry-run-init-env \
	"$@" # script arguments (can override args above)