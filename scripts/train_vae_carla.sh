#!/usr/bin/env bash

ENV="Carla-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/VAE/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/VAE"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

python main.py \
	--mode train --gpu 1 \
	--env "$ENV" \
    --n-past-actions 8 \
	--hidden-dims 256 128 \
	--activation LeakyReLU \
	--encoder-arch VAE \
	--weight-path "/root/thesis/thesis-code/Soft-Actor-Critic/vae_weights/Carla-v0/epoch(10)-loss(+2.431E+05).pkl" \
	--state-dim 1024 \
	--max-episode-steps 999 \
	--n-epochs 1000 --n-updates 256 --batch-size 16 \
	--n-samplers 1 \
	--buffer-capacity 2500 \
	--update-sample-ratio 5.0 \
	--critic-lr 1E-4 --actor-lr 1E-4 \
	--alpha-lr 1E-4 --initial-alpha 1.0 --adaptive-entropy \
	--normalize-rewards --reward-scale 1.0 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--dry-run-init-env \
	--load-checkpoint \
	"$@" # script arguments (can override args above)
