#!/usr/bin/env bash

ENV="CarRacing-v0"
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

PYTHONWARNINGS=ignore xvfb-run -s "-screen 0 1400x900x24" python main.py \
	--mode train --gpu 3 \
	--env "$ENV" \
	--vision-observation --image-size 96 \
	--n-frames 4 \
    --repeat-action 4 \
	--hidden-dims 256 128 \
	--activation LeakyReLU \
	--encoder-arch CNN \
	--state-dim 256 \
	--encoder-hidden-channels 8 16 32 64 128 256 \
	--kernel-sizes 4 3 3 3 3 3 \
	--strides 2 2 2 2 1 1 \
	--paddings 0 0 0 0 0 0 \
	--encoder-activation ReLU \
	--max-episode-steps 999 \
	--n-epochs 1000 --n-updates 256 --batch-size 64 \
	--n-samplers 1 \
	--buffer-capacity 10000 \
	--update-sample-ratio 8.0 \
	--critic-lr 1E-4 --actor-lr 1E-5 \
	--alpha-lr 1E-5 --initial-alpha 1.0 --adaptive-entropy \
	--normalize-rewards --reward-scale 1.0 \
	--gamma 0.99 --soft-tau 0.01 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	"$@" # script arguments (can override args above)
