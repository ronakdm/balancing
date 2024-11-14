#!/bin/bash

model="orig_clip"
model_type="miniclip"
root="~" # Where you would like your datasets to be downloaded.
device="cuda:0"

for dataset in mscoco_captions flickr8k;
do
clip_benchmark eval --dataset=$dataset \
    --task=zeroshot_retrieval \
    --pretrained='laion2b_s34b_b79k' \
    --device=$device \
    --model=$model \
    --model_type=$model_type \
    --output="zeroshot_retrieval_{$dataset}_${ckpt}.json" \
    --dataset_root=$root \
    --batch_size=64 
done