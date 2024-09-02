#!/usr/bin/env bash
gpuid=2
dataset='aircraft_fs'
model='PDNet16'
method='protonet'
check_resume='False'
check_epoch=0

# 预训练超参：
pre_optim='SGD'   #['SGD','Adam']
pre_batch_size=64
pre_lr=0.05
pre_epoch=170
pre_num_episode=50

# 元训练超参：
optim='SGD'   #['SGD','Adam']
meta_lr=0.0001
meta_epoch=80
meta_train_episode=600
meta_val_episode=600


DATA_ROOT=/home/ssdData/qcfData/BPIAL_benchmark/$dataset

MODEL_PATH=./checkpoints/$dataset/Mono_${model}_${method}/prephase_${pre_optim}_${pre_batch_size}_${pre_lr}_${pre_num_episode}_${pre_epoch}/best_model.tar
cd ../


echo "============= meta-train 5-shot ============="
python mono2_metatrain.py --pre_batch_size $pre_batch_size --pre_epoch $pre_epoch --pre_num_episode $pre_num_episode --dataset $dataset --data_path $DATA_ROOT --model $model --method $method \
--pre_optimizer $pre_optim --check_resume $check_resume --check_epoch $check_epoch --optimizer $optim --image_size 84 --gpu ${gpuid} --pre_lr $pre_lr --meta_lr $meta_lr \
--epoch $meta_epoch --milestones 40 80 --n_shot 5 --train_n_episode $meta_train_episode --val_n_episode $meta_val_episode --pretrain_path $MODEL_PATH