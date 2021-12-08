# Scale comparison for HM3D
We compare the scale of HM3D to other datasets using a number of metrics that measure the overall
floor area, navigable area, and structural complexity of the scenes. We provide instructions to compute metrics from Table 1 in the [HM3D paper](https://openreview.net/pdf?id=-v4OuqNs5P).

1. **Testing:** Download the habitat test scenes and run the metric computation code.
    ```
    conda activate hm3d

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


2. **Reproducing results:** Ensure that the environment variables corresponding to the different datasets are set as instructed in the main README. Execute the run script.
    ```
    chmod +x run.sh && ./run.sh
    ```
