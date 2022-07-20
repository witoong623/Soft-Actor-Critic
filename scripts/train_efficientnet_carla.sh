#!/usr/bin/env bash

ENV="Carla-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/EFFICIENTNET/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/EFFICIENTNET"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

# more random spawn in route, float32 training
# perfect sample action at start

python main.py \
	--mode train --gpu 0 --sampler-gpu 1 \
	--env "$ENV" \
	--n-frames 2 \
    --n-past-actions 10 \
	--image-size 256 512 \
	--camera-size 600 800 \
	--camera-fov 110 \
	--fps-mode low \
	--encoder-arch EFFICIENTNET \
	--activation LeakyReLU \
	--state-dim 1024 \
	--hidden-dims 512 256 \
	--max-episode-steps 10000 \
	--n-epochs 1000 --n-updates 256 --batch-size 32 \
	--n-samplers 1 \
	--buffer-capacity 60000 \
	--update-sample-ratio 2.0 \
	--critic-lr 3E-4 --actor-lr 3E-4 \
	--alpha-lr 3E-4 --initial-alpha 1.0 \
	--adaptive-entropy --target-entropy -3 \
	--n-bootstrap-step 2 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--log-episode-video \
	--dry-run-init-env \
	"$@" # script arguments (can override args above)
