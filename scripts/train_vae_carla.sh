#!/usr/bin/env bash

ENV="Carla-v0"
DATETIME="$(date +"%Y-%m-%d-%T")"
LOG_DIR="logs/$ENV/BETAVAE/$DATETIME"
CHECKPOINT_DIR="savedcheckpoints/$ENV/BETAVAE"

ROOT_DIR="$(
	cd "$(dirname "$(dirname "$0")")"
	pwd
)"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR"

# -0.1 every time step, more random spawn in route, float32 training, 5FPS
# perfect sample action at start

python main.py \
	--mode train --gpu 0 --sampler-gpu 1 \
	--env "$ENV" \
    --n-past-actions 10 \
	--image-size 256 512 \
	--camera-fov 110 \
	--camera-size 600 800 \
	--hidden-dims 512 256 \
	--activation LeakyReLU \
	--encoder-arch BETAVAE \
	--weight-path "vae_weights/Carla-v0_town7_b3_new_tanh_mse/bvae_town7_epoch(100)-loss(+8.729E+04).pkl" \
	--state-dim 512 \
	--max-episode-steps 5000 \
	--n-epochs 1000 --n-updates 256 --batch-size 64 \
	--n-samplers 1 \
	--buffer-capacity 23000 \
	--update-sample-ratio 2.0 \
	--critic-lr 1E-4 --actor-lr 1E-4 \
	--alpha-lr 1E-4 --initial-alpha 1.0 --adaptive-entropy \
	--target-entropy -3 \
	--n-bootstrap-step 3 \
	--gamma 0.99 --soft-tau 0.005 --random-seed 69 \
	--log-dir "$LOG_DIR" \
	--checkpoint-dir "$CHECKPOINT_DIR" \
	--log-episode-video \
	--dry-run-init-env \
	"$@" # script arguments (can override args above)
