#!/usr/bin/env bash

ENV="Pendulum-v0"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"

python3 main.py --mode train --gpu 0 --env "$ENV" \
	--net RNN --activation LeakyReLU \
	--hidden-dims-before-lstm 256 \
	--hidden-dims-lstm 128 \
	--hidden-dims-after-lstm 128 \
	--skip-connection \
	--state-dim 256 \
	--encoder-hidden-dims 256 256 128 128 \
	--max-episode-steps 500 \
	--n-epochs 4000 --n-updates 256 --batch-size 16 --step-size 16 \
	--n-samplers 4 \
	--buffer-capacity 1000000 \
	--update-sample-ratio 2.0 \
	--soft-q-lr 1E-3 --policy-lr 1E-4 --alpha-lr 1E-3 \
	--initial-alpha 1.0 --auto-entropy \
	--weight-decay 1E-5 --random-seed 0 \
	--log-dir "logs/$ENV/RNN" \
	--checkpoint-dir "checkpoints/$ENV/RNN"
