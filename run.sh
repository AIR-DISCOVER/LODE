# GPU_COUNT=4
GPU_COUNT=1

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --nnodes=1 --node_rank=0 --master_port=23456 main_sc.py --task=train --experiment_name=test
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --nnodes=1 --node_rank=0 --master_port=23456 main_sc.py --task=valid --experiment_name=test
