#!/usr/bin/env bash
gpuid=2
dataset='cifar_fs'
model='CCAFNet'
method='protonet'


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

MODEL_1SHOT_PATH=./checkpoints/$dataset/Mono_${model}_${method}/metaphase_${pre_optim}_${pre_batch_size}_${pre_lr}_${pre_num_episode}_${pre_epoch}_5way_1shot/metatrain_${optim}_${meta_lr}_${meta_epoch}_train_${meta_train_episode}_val_${meta_val_episode}/best_model.tar
MODEL_5SHOT_PATH=./checkpoints/$dataset/Mono_${model}_${method}/metaphase_${pre_optim}_${pre_batch_size}_${pre_lr}_${pre_num_episode}_${pre_epoch}_5way_5shot/metatrain_${optim}_${meta_lr}_${meta_epoch}_train_${meta_train_episode}_val_${meta_val_episode}/best_model.tar
cd ../


echo "============= meta-test 5-shot ============="
N_SHOT=5
python mono3_metatest.py --pre_epoch $pre_epoch --pre_num_episode $pre_num_episode --meta_train_n_episode $meta_train_episode --meta_val_n_episode $meta_val_episode --meta_epoch $meta_epoch --pre_batch_size $pre_batch_size --pre_lr $pre_lr --meta_lr $meta_lr --dataset $dataset --data_path $DATA_ROOT --model $model --method $method \
--pre_optimizer $pre_optim --optimizer $optim --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --model_path $MODEL_5SHOT_PATH --test_task_nums 3 --test_n_episode 1000
