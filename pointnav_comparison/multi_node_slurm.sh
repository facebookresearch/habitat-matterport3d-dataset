#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=train_hm3d_seed_100
#SBATCH --output=logs.ddppo_%j.out
#SBATCH --error=logs.ddppo_%j.err
#SBATCH --gres=gpu:2
#SBATCH --nodes=16
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=2
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=<partition>

# Turn off non-critical habitat-sim logging
export GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR
export MASTER_PORT=8537

ROOT_DIR=$PWD

# Load necessary software using "module". Alternatively, the paths to each 
# of the dependencies can be added manually.
module purge
module load anaconda3/2020.11
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load gcc/7.3.0
module load cmake/3.15.3/gcc.7.3.0

conda activate hm3d

export NCCL_SOCKET_IFNAME=""
export GLOO_SOCKET_IFNAME=""

cd <PATH TO habitat-lab directory>

set -x
srun python -u -m habitat_baselines.run \
    --exp-config $ROOT_DIR/ddppo_train.yaml \
    --run-type train \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 50000