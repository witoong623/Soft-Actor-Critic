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
	--mode train --gpu 0 \
	--env "$ENV" \
	--vision-observation --image-size 96 \
	--n-frames 4 \
    --n-past-actions 8 \
	--hidden-dims 256 128 \
	--activation LeakyReLU \
	--encoder-arch EFFICIENTNET \
	--state-dim 1408 \
	--max-episode-steps 999 \
	--n-epochs 1000 --n-updates 256 --batch-size 64 \
	--n-samplers 1 \
	--buffer-capacity 5000 \
	--update-sample-ratio 8.0 \
	--critic-lr 3E-4 --actor-lr 3E-4 \
	--alpha-lr 3E-4 --initial-alpha 1.0 --adaptive-entropy \
	--normalize-rewards --reward-scale 1.0 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	"$@" # script arguments (can override args above)
