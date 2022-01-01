python main.py \
	--mode test_render \
	--env Carla-v0 \
	--gpu 0 \
	--n-past-actions 10 \
	--hidden-dims 256 128 \
	--activation LeakyReLU \
	--encoder-arch EFFICIENTNET \
	--state-dim 1280 \
	--max-episode-steps 999 \
	--random-seed 20 \
	--record-video \
	--load-checkpoint \
	--dry-run-init-env \
	--log-dir "logs/Carla-v0_test_render/EFFICIENTNET" \
	--checkpoint-dir "savedcheckpoints/Carla-v0/EFFICIENTNET"
