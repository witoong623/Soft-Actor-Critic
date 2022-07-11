#!/usr/bin/env bash


python -m extra.record_agent_video \
	--env "Carla-v0" \
	--image-size 256 512 \
	--camera-size 600 800 \
	--camera-fov 110 \
	--traffic-mode LHT \
	--fps-mode high \
	--max-episode-steps 1000 \
	--record-video \
	--video-dir "carla_videos/carla_agent_videos" \
	--dry-run-init-env \
	"$@" # script arguments (can override args above)
