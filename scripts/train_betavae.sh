torchrun \
	--standalone \
	--nnodes=1 \
	--nproc_per_node=2 \
	vae/train.py
