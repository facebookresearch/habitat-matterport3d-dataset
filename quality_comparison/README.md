# Reconstruction quality comparison for HM3D
We provide instructions to compute the reconstruction completeness and visual fidelity metrics in the [HM3D paper](https://openreview.net/pdf?id=-v4OuqNs5P) (Fig. 4 and Table 5, respectively).

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

Install remaining requirements from pip.
```
pip install -r requirements.txt
```

## Reproducing results from the HM3D paper
In our paper, we compared HM3D to Gibson, MP3D, RoboThor, Replica, and ScanNet. Download each dataset based on [instructions](https://github.com/facebookresearch/habitat-sim/blob/master/DATASETS.md) from habitat-sim. In the case of RoboThor, convert the raw scan assets to GLB using [assimp](https://github.com/assimp/assimp).

```
assimp export <SOURCE SCAN FILE> <GLB FILE PATH>
```
We measure the reconstruction quality of a 3D scene by rendering RGB-D images from the scene using Habitat.
To measure reconstruction completeness, we define a view-based metric that measures the degree to which reconstruction artifacts (or defects) occur in an image. Reconstruction artifacts include missing surfaces, holes, and untextured surface regions. We then quantify the fraction of viewpoints in the scene that have a significant number of defects (i.e., the `%defects` metric).
To measure visual fidelity, we compare the rendered RGB images with real-world captured images generated from high-resolution panoramas in Gibson and MP3D using divergence metrics such as KID and FID.

### Simulated image extraction
Once all datasets are download and processed, update paths to the glb / ply files and `SAVE_DIR_PATH` (any directory to save extracted images) in `run_sim_extraction.sh`. We render simulated images for each dataset by sampling locations in a uniform grid of navigable locations.

```
chmod +x run_sim_extraction.sh && ./run_sim_extraction.sh
```

### Real-world image extraction
We extract real-world images by randomly sampling perspective images from the raw 360 panoramas released in Gibson and MP3D.  Update the path to raw gibson dataset containing the panoramas as `GIBSON_DIR` in `run_real_extraction.sh`. Obtain the 360 panorama images for MP3D using the [PanoBasic toolkit](https://github.com/yindaz/PanoBasic/blob/master/demo_matterport.m). Update the path to the MP3D panoramas as `MP3D_PANO_DIR` in `run_real_extraction.sh`. Run the extraction script.

```
chmod +x run_real_extraction.sh && ./run_real_extraction.sh
```

### Measuring visual fidelity
Update `SAVE_DIR_PATH` in `run_visual_fidelity.sh` (same path as before) and execute it.
```
chmod +x run_visual_fidelity.sh && ./run_visual_fidelity.sh
```

### Measuring reconstruction completeness
Update `SAVE_DIR_PATH` in `run_reconstruction_completeness.sh` (same path as before) and execute it.
```
chmod +x run_reconstruction_completeness.sh && ./run_reconstruction_completeness.sh
```
