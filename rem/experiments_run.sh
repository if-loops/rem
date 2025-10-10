#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file is used to run the experiments from the paper.
# It is recommended to run it from the rem folder of the repository.
# For tracking purposes
VERSION_I="version"
project_name_prefix="prefix"
#-----------------
# General settings
seeds=(0 1 2)
rn_dataset="CIFAR10"
rn_num_classes=10
rn_model="resnet9"
rn_pretrain_iters=4000
rn_unlearn_iters=980
rn_pretrain_lr=0.025
rn_unlearn_lr=0.005
rn_batch_size=512
vit_dataset="SVHN"
vit_num_classes=10
vit_model="ViT"
vit_pretrain_iters=1440 # different batch size (same epoch number)
vit_unlearn_iters=360 # different batch size (same epoch number)
vit_pretrain_lr=0.0001 # 1e-4 as typical transformer lr
vit_unlearn_lr=0.00002 # same ratio as for RN
vit_batch_size=2048
ti_dataset="tinyimagenet"
ti_dataset="Imagenette"
ti_num_classes=10
ti_model="resnet9"
ti_pretrain_iters=800
ti_unlearn_iters=200
ti_pretrain_lr=0.025
ti_unlearn_lr=0.005
ti_batch_size=512
dataset_methods=("poisoning" "interclasslabelswap" "randomlabelswap")
forget_set_sizes=(1000)
deletion_sizes=(1000 100 900 800 700 600 500 400 300 200)
# From corrective unlearning / potion repos (do not change)
patch_size=3
p_cuts=(0.20) # Df unlearning threshold from potion
# Setting the pretraining type
'''
Combine the following building blocks to create the desired training recipe
Naive...Normal training with 100% model capacity
ETD...Start with ETD to train an ETD model or a capacity restriced model
ETDfix__...Add the desired capacity of the model at inference as an integer (e.g., 50)
ETDfix50mem__...Add the desired size of each ETD mask relative to 100%-fix. (e.g. 20). Use 0 for no ETD, i.e. a capacity restriced model.
ETDfix50mem20drop_...Set the way the model should be used at inference/test time. 1: only use the generalization part (typical ETD usage); 0: Use full model capacity; 2: Give ETD full knowledge of Df for masking and use REM masking strategy during ETD training
'''
example_experiments=("ETDfix50mem20drop1" "ETDfix50mem10drop1" "ETDfix50mem5drop1" "ETDfix50mem0drop1" "ETDfix90mem20drop1" "ETDfix90mem10drop1" "ETDfix90mem5drop1" "ETDfix90mem20drop1")
core_experiments=("ETDfix50mem0drop1" "ETDfix50mem20drop1")
train_types=("${core_experiments[@]}")
#################################################################
############ Only modify below to add/remove methods ############
#################################################################
#-----------------
setups=("ResNet" "ViT") # outer loop runs both architectures
# set values for ResNet
dataset=$rn_dataset
model=$rn_model
pretrain_iters=$rn_pretrain_iters
unlearn_iters=$rn_unlearn_iters
pretrain_lr=$rn_pretrain_lr
num_classes=$rn_num_classes
unlearn_lr=$rn_unlearn_lr
for setup in "${setups[@]}"; do
  if [[ "$setup" == "ResNet" ]]; then
    echo "==> ResNet"
    dataset=$rn_dataset
    model=$rn_model
    pretrain_iters=$rn_pretrain_iters
    unlearn_iters=$rn_unlearn_iters
    pretrain_lr=$rn_pretrain_lr
    num_classes=$rn_num_classes
    unlearn_lr=$rn_unlearn_lr
    batch_size=$rn_batch_size
    optim="SGD"
  else
    echo "==> ViT"
    dataset=$vit_dataset
    model=$vit_model
    pretrain_iters=$vit_pretrain_iters # different batch size (same epoch number)
    unlearn_iters=$vit_unlearn_iters # different batch size (same epoch number)
    pretrain_lr=$vit_pretrain_lr
    num_classes=$vit_num_classes
    unlearn_lr=$vit_unlearn_lr
    batch_size=$vit_batch_size
    optim="ADAM"
  fi
  for seed in "${seeds[@]}"; do
    for p_cut in "${p_cuts[@]}"; do
      project_name=$project_name_prefix+$VERSION_I+$p_cut
      echo $project_name
      for forget_set_size in "${forget_set_sizes[@]}"; do
          rm -r results.tsv
          for train_type in "${train_types[@]}"; do
              for deletion_size in "${deletion_sizes[@]}"; do
                for dataset_method in "${dataset_methods[@]}"; do
                    #rm -r logs/*
                    rm -r results.tsv
                    # SalUn
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=SalUn --unlearn_iters=$unlearn_iters
                    #################################################################
                    # --- REM ---
                    #REM (our method)
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=REM --unlearn_iters=$unlearn_iters
                    #REMIDEAL (has perfect knowledge of poison IDs for masking)
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=REMIDEAL --unlearn_iters=$unlearn_iters
                    #################################################################
                    # --- Becnhmarks ---
                    # SCRUB
                    unlearn_iters_=($unlearn_iters)
                    alpha_=(0.001 0.01 0.1 10)
                    unlearn_lr_=($unlearn_lr)
                    for unlearn_iters_scrub in "${unlearn_iters_[@]}"; do
                      for alpha_scrub in "${alpha_[@]}"; do
                        for unlearn_lr_scrub in "${unlearn_lr_[@]}"; do
                          echo $unlearn_iters_scrub
                          echo $alpha_scrub
                          echo $unlearn_lr_scrub
                          python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=Scrub --unlearn_iters=$unlearn_iters_scrub --msteps=100 --unlearn_lr=$unlearn_lr_scrub --alpha=$alpha_scrub
                        done
                      done
                    done
                    # Retraining from scratch
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
                    # Ascent
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=Ascent --unlearn_iters=$unlearn_iters
                    # Potion
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=XALFSSD
                    # BadT
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=BadT --unlearn_iters=$unlearn_iters
                    # Finetuning on Dr
                    #python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=CF --unlearn_iters=$unlearn_iters --k=-1
                    #################################################################
                    # --- Ablations ---
                    # NPO: No Step 3.1 and 3.2 (i.e. no step 3) = NPO (as a baseline just the removal step)
                    python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=NPO --unlearn_iters=$unlearn_iters
                    # No Step 3.1 "REMGEN" (Ablation: does not use redirection/mem part of step 3.1)
                    #python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=REMGEN --unlearn_iters=$unlearn_iters
                    # No Step 3.2 "NPOREMMOD" (Ablation: does not use removal during redirection with step 3.2)
                    #python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=NPOREMMOD --unlearn_iters=$unlearn_iters
                    # ----
                    # REMMOD is REM but with gradient ascent instead of NPO
                    python src/main.py --batch_size=$batch_size --optim=$optim --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=REMMOD --unlearn_iters=$unlearn_iters
                    # REMMOD IDEAL (has perfect knowledge of poison IDs for masking)
                    #python src/main.py --batch_size=$batch_size --optim=$optim --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=REMMODIDEAL --unlearn_iters=$unlearn_iters
                done
            done
        done
      done
    done
  done
done
#-----------------
""" # commented out as only used for appendix, preventing users from mistakenly running longer than necessary experiments
# ETD comparisons to restricted models
exp_list=("ETDfix90mem20drop1" "ETDfix90mem0drop1" "ETDfix50mem20drop1" "ETDfix50mem0drop1" "ETDfix25mem20drop1" "ETDfix25mem0drop1")
train_types=("${exp_list[@]}")
forget_set_sizes=(1000)
deletion_sizes=(1000) # only running the full size
#-----------------
for seed in "${seeds[@]}"; do
  for dataset_method in "${dataset_methods[@]}"; do
    for p_cut in "${p_cuts[@]}"; do
      project_name=$project_name_prefix+$VERSION_I+$p_cut
      echo $project_name
      for forget_set_size in "${forget_set_sizes[@]}"; do
          rm -r results.tsv
          for train_type in "${train_types[@]}"; do
              for deletion_size in "${deletion_sizes[@]}"; do
                  #rm -r logs/*
                  rm -r results.tsv
                  # EU
                  python src/main.py --batch_size=$batch_size --optim=$optim --seed=$seed --unlearn_lr=$unlearn_lr --pretrain_lr=$pretrain_lr --train_type=$train_type --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
              done
          done
      done
    done
  done
done
#-----------------
"""
# aggregate all the results into "small_table.csv" and create the plots
python plotting.py
# create LateX ready tables
python tables.py
