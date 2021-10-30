#!/usr/bin/env bash

ENV="Pendulum-v1"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/RNN/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/RNN"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

PYTHONWARNINGS=ignore python3 main.py \
	--mode train --gpu 0 1 2 1 2 \
	--env "$ENV" \
	--hidden-dims 128 64 \
	--activation LeakyReLU \
	--encoder-arch RNN \
	--state-dim 128 \
	--encoder-hidden-dims-before-rnn 128 \
	--encoder-hidden-dims-rnn 64 \
	--encoder-hidden-dims-after-rnn 128 \
	--encoder-activation LeakyReLU \
	--skip-connection \
	--max-episode-steps 500 \
	--n-epochs 1000 --n-updates 256 --batch-size 64 --step-size 8 \
	--n-samplers 4 \
	--buffer-capacity 1000000 \
	--update-sample-ratio 2.0 \
	--critic-lr 1E-3 --actor-lr 1E-4 \
	--alpha-lr 1E-3 --initial-alpha 1.0 --adaptive-entropy \
	--gamma 0.99 --soft-tau 0.01 \
	--normalize-rewards --reward-scale 1.0 \
	--weight-decay 1E-5 --random-seed 0 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	"$@" # script arguments (can override args above)
