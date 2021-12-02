# Scale comparison for HM3D
We provide instructions to compute metrics from Table 1 in the [HM3D paper](https://openreview.net/pdf?id=-v4OuqNs5P).

## Environment setup
Create conda environment and activate it.
```
conda env create -n hm3d python=3.8.3
conda activate hm3d
```

Install the latest habitat-sim.
```
conda install habitat-sim headless -c conda-forge -c aihabitat
```

Install trimesh with soft dependencies.
```
pip install "trimesh[easy]"
```

Install remaining requirements from pip.
```
pip install -r requirements.txt
```

## Testing

To test the scale comparison code, download the habitat test scenes and run the metric computation code.
```
python -m habitat_sim.utils.datasets_download \
    --uids habitat_test_scenes \
    --data-path data

python compute_scene_metrics.py \
    --dataset-root data/scene_datasets/habitat-test-scenes \
    --save-path data/test_metrics.csv
```

The expected outputs are:
```
============= Metrics =============
Number of scenes: 3
navigable_area                 | 288.710
navigation_complexity          | 9.466
scene_clutter                  | 2.978
floor_area                     | 512.045
```

We tested these instructions on MacOS and Ubuntu 20.04.


## Reproducing results from the HM3D paper
In our paper, we compared HM3D to Gibson, MP3D, RoboThor, Replica, and ScanNet. Download each dataset based on [instructions](https://github.com/facebookresearch/habitat-sim/blob/master/DATASETS.md) from habitat-sim. In the case of RoboThor, convert the raw scan assets to GLB using [assimp](https://github.com/assimp/assimp).
```
assimp export <SOURCE SCAN FILE> <GLB FILE PATH>
```
Once the datasets are download and processed, update paths to the glb / ply files in `run.sh` and execute it.
```
chmod +x run.sh && ./run.sh
```
