#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_CA_mdetr_data
MDETR_DATA=./data/mdetr_data

PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --backbone resnet101 --coco_path ${MDETR_DATA} ${PY_ARGS}
