#!/bin/bash
#SBATCH --job-name=e_hm3d_depth_seed_100
#SBATCH --array=0-2
#SBATCH --output=eval_logs.ddppo_%A_%a.out
#SBATCH --error=eval_logs.ddppo_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=<partition>

# Turn off non-critical habitat-sim logging
export GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet

ROOT_DIR=$PWD

# Load necessary software using "module". Alternatively, the paths to each 
# of the dependencies can be added manually.
module purge
module load anaconda3/2020.11
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load gcc/7.3.0
module load cmake/3.15.3/gcc.7.3.0

cd <PATH TO habitat-lab directory>


if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]; then

  echo "Evaluating gibson"
  set -x
  python -u habitat_baselines/run.py \
      --exp-config $ROOT_DIR/ddppo_eval_gibson.yaml \
      --run-type eval

elif [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then

  echo "Evaluating mp3d"
  set -x
  python -u habitat_baselines/run.py \
      --exp-config $ROOT_DIR/ddppo_eval_mp3d.yaml \
      --run-type eval

else

  echo "Evaluating hm3d"
  set -x
  python -u habitat_baselines/run.py \
      --exp-config $ROOT_DIR/ddppo_eval_hm3d.yaml \
      --run-type eval

fi
