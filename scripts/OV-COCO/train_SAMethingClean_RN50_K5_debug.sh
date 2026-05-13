#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_COCO/SAMethingClean_RN50_K5.py \
	--amp \
	--device $device \
	--amp \
	--eval_start_epoch 1 \
	--eval_every_epoch 2 \
	--debug \
