# Utility of HM3D for Embodied AI
A popular downstream application for large-scale 3D reconstruction datasets has been to use them
with 3D simulation platforms to study embodied AI tasks such as visual navigation. Since HM3D improves over existing datasets both in terms of size and quality, we evaluate its utility for Embodied AI by training PointNav agents. We provide instructions for running the PointNav experiments reported in Tab. 2 from the HM3D paper.
We use existing code from [Habitat baselines](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines) to perform our experiments.

## Setup instructions
* Follow instructions from [habitat-lab repository](https://github.com/facebookresearch/habitat-lab) to install habitat-lab and baselines v0.2.1.
* Follow instructtions from [habitat-sim repository](https://github.com/facebookresearch/habitat-sim) to install habitat-sim v0.2.1 with headless rendering and cuda support.
* Download the scene datasets for Gibson, MP3D and HM3D as instructed [here](https://github.com/facebookresearch/habitat-lab#data).
* Download PointNav-v1 task datasets for Gibson, Gibson 0+ train, MP3D, and HM3D from [here](https://github.com/facebookresearch/habitat-lab#task-datasets).


## Running experiments
We use [DD-PPO](https://arxiv.org/abs/1911.00357) implementation in [habitat_baselines](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines) for our experiments.  In this directory, we provide the train and evaluation configs and scripts used for our experiments. This specific example corresponds to training on HM3D (train), and evaluating on Gibson (val), HM3D (val) and MP3D (val).

### Config files
These are the config files for training on HM3D and evaluating on other datasets.
* Train on HM3D (train): `ddppo_train.yaml`
* Evaluate on Gibson (val): `ddppo_eval_gibson.yaml`
* Evaluate on MP3D (val): `ddppo_eval_mp3d.yaml`
* Evaluate on HM3D (val): `ddppo_eval_hm3d.yaml`

### Training script
A slurm submission script for distributed training is provided in `multi_node_slurm.sh`. Each experiment is run in an internal cluster with 8 nodes, and 4 Volta 16/32GB GPUs per node. The experiment is scheduled as follows.

```
sbatch multi_node_slurm.sh
```
To train on any other dataset `<dataset>`, change `ddppo_train.yaml` as follows:
* Set `BASE_TASK_CONFIG_PATH` to `configs/tasks/pointnav_<dataset>.yaml`
* Replace directory paths with `hm3d-depth`  to `<dataset>-depth`

To train with the RGB sensor, change `ddppo_train.yaml` as follows:
* Change `SENSORS: ["DEPTH_SENSOR"]` to `SENSORS: ["RGB_SENSOR"]`
* Change directory paths with `<dataset>-depth` to `<dataset>-rgb`

## Evaluation script
A slurm submission script for evaluating each saved checkpoint is provided in `submit_eval.sh`. Evaluation is performed in an internal cluster with 3 Volta 16/32GB GPUs (each dataset evaluation is performed on a different GPU). The evaluation is scheduled as follows.

```
sbatch submit_eval.sh
```

To evaluate on a specific checkpoint `<CKPT DIR>/ckpt.<CKPT ID>.pth`, update `ddppo_eval_*.yaml`:
* Set `EVAL_CKPT_PATH_DIR` to `<CKPT DIR>/ckpt.<CKPT ID>.pth`

To evaluate on the test split, update `ddppo_eval_*.yaml`:
* Change `SPLIT: 'val'` to `SPLIT: 'test'`

To evaluate a model with RGB sensor, update `ddppo_eval_*.yaml`:
* Change `SENSORS: ["DEPTH_SENSOR"]` to `SENSORS: ["RGB_SENSOR"]`

## Pre-trained models
We provide the saved checkpoints that can be used to reproduce results in Table 2 and Figure 7 [here](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/hm3d_ddppo_pointnav_baselines_v1.zip). We select the checkpoint to evaluate based on the validation curves. For each model, we pick the checkpoint with the highest Gibson (val) performance.
