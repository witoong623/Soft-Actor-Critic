#!/usr/bin/env bash

ENV="Carla-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/CNN/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/CNN"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

python main.py \
	--mode train --gpu 0 --sampler-gpu 1 \
	--env "$ENV" \
	--n-frames 2 \
    --n-past-actions 10 \
	--image-size 256 512 \
	--camera-size 600 800 \
	--camera-fov 110 \
	--fps-mode high \
	--encoder-arch CNN \
	--encoder-hidden-channels 64 128 256 \
	--activation ELU \
	--encoder-activation ELU \
	--state-dim 768 \
	--hidden-dims 512 256 \
	--max-episode-steps 5000 \
	--n-epochs 1000 --n-updates 256 --batch-size 32 \
	--n-samplers 1 \
	--buffer-capacity 21000 \
	--update-sample-ratio 2.0 \
	--critic-lr 3E-4 --actor-lr 3E-4 \
	--alpha-lr 3E-4 --initial-alpha 1.0 \
	--adaptive-entropy --target-entropy -3 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--n-bootstrap-step 2 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--checkpoint-save-frequency 5 \
	--log-episode-video \
	--dry-run-init-env \
	"$@" # script arguments (can override args above)
