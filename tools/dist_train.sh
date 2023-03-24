#!/usr/bin/env bash

VISIBLE_GPU=0,1
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=${VISIBLE_GPU} \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    --data-cfg config/dataset/coco.yaml \
    --model-cfg config/model/yolov5s_friendly.yaml \
    --weights '' \
    --batch-size 64 \
    --epochs 300
