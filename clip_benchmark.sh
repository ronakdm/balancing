#!/bin/bash

model="orig_clip"
model_type="miniclip"
root="~" # Where you would like your datasets to be downloaded.
device="cuda:0"

for dataset in cifar10 cifar100 stl10;
do
    clip_benchmark eval --dataset=$dataset \
    --task=zeroshot_classification \
    --pretrained='laion2b_s34b_b79k' \
    --device=$device \
    --model=$model \
    --model_type=$model_type \
    --output="${dataset}_${model}.json" \
    --dataset_root=$root \
    --batch_size=64
done