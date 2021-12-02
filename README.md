# Habitat-Matterport 3D Dataset (HM3D)

The Habitat-Matterport 3D Research Dataset is the largest-ever dataset of 3D indoor spaces. It consists of 1,000 high-resolution 3D scans (or digital twins) of building-scale residential, commercial, and civic spaces generated from real-world environments.

HM3D is free and [available now](https://matterport.com/habitat-matterport-3d-research-dataset) for academic, non-commercial research. Researchers can use it with FAIR’s [Habitat simulator](https://aihabitat.org/) to train embodied agents, such as home robots and AI assistants, at scale.

![example](./assets/HM3D-intro.png)


This repository contains the code required to reproduce experiments from our NeurIPS 2021 paper. If you use the HM3D dataset or the experimental code in your research, please cite the [HM3D](https://openreview.net/pdf?id=-v4OuqNs5P) paper. Please check out our [website](https://aihabitat.org/datasets/hm3d/) for details on downloading and visualizing the HM3D dataset.

```
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=-v4OuqNs5P}
}
```

## Installation instructions
We provide a common set of instructions to setup the environment to run all our experiments.
1. Clone the HM3D github repository and add it to `PYTHONPATH`.
    ```
    git clone https://github.com/facebookresearch/habitat-matterport3d-dataset.git
    cd habitat-matterport3d-dataset
    export PYTHONPATH=$PYTHONPATH:$PWD
    ```
2. Create conda environment and activate it.
    ```
    conda env create -n hm3d python=3.8.3
    conda activate hm3d
    ```
3. Install habitat-sim using conda.
    ```
    conda install habitat-sim headless -c conda-forge -c aihabitat
    ```
    See habitat-sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

4. Install trimesh with soft dependencies.
    ```
    pip install "trimesh[easy]"
    ```

5. Install remaining requirements from pip.
    ```
    pip install -r requirements.txt
    ```

## Downloading datasets
In our paper, we compared HM3D to Gibson, MP3D, RoboThor, Replica, and ScanNet. 

* Download each dataset based on these [instructions](https://github.com/facebookresearch/habitat-sim/blob/master/DATASETS.md) from habitat-sim. In the case of RoboThor, convert the raw scan assets to GLB using [assimp](https://github.com/assimp/assimp).
  ```
  assimp export <SOURCE SCAN FILE> <GLB FILE PATH>
  ```

* Once the datasets are download and processed, create environment variables pointing to the corresponding scene paths.
  ```
  export GIBSON_ROOT=<PATH TO GIBSON glbs>
  export MP3D_ROOT=<PATH TO MP3D glbs>
  export ROBOTHOR_ROOT=<PATH TO ROBOTHOR glbs>
  export HM3D_ROOT=<PATH TO HM3D glbs>
  export REPLICA_ROOT=<PATH TO REPLICA plys>
  export SCANNET_ROOT=<PATH TO SCANNET glbs>
  ```


## Running experiments
We provide the code for reproducing the results from [our paper](https://openreview.net/pdf?id=-v4OuqNs5P) in different directories.
* `scale_comparison` contains the code for comparing the scale of HM3D with other datasets (Tab. 1 in the paper). 
* `quality_comparison` contains the code for comparing the reconstruction completeness and visual fidelity of HM3D with other datasets (Fig. 4 and Tab. 5 in the paper). 
* `pointnav_comparison` contains the configs and instructions to train and evaluate PointNav agents on HM3D and other datasets (Tab. 2 and Fig. 7 in the paper).

We further provide README files within each directory with instructions for running the corresponding experiments.
