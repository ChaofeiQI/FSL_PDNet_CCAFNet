#!/usr/bin/env bash
gpuid=1
dataset='cub_cropped'
model='PDNet19'
method='protonet'

check_resume='False'
check_epoch=0

# 预训练超参：
optim='SGD'   #['SGD','Adam']
pre_batch_size=64
pre_lr=0.05
pre_epoch=170
pre_num_episode=50


DATA_ROOT=/home/ssdData/qcfData/BPIAL_benchmark/$dataset
cd ../


echo "============= pre-train ============="
python mono1_pretrain.py  --batch_size $pre_batch_size --dataset $dataset --data_path $DATA_ROOT --model $model --method $method \
--check_resume $check_resume --check_epoch $check_epoch --optimizer $optim --image_size 84 --gpu ${gpuid} --pre_lr $pre_lr --wd 1e-4 \
--epoch $pre_epoch --milestones 100 150 --save_freq 100 --val meta --val_n_episode $pre_num_episode --n_shot 5
