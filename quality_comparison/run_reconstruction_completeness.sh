#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MD_SAVE_ROOT="$SAVE_DIR_PATH/simulated_images/rgb2depth_metadata"

python measure_reconstruction_completeness.py \
    --json-paths $MD_SAVE_ROOT/gibson_4_plus_sim_metadata.json \
                $MD_SAVE_ROOT/gibson_sim_metadata.json \
                $MD_SAVE_ROOT/robothor_sim_metadata.json \
                $MD_SAVE_ROOT/mp3d_sim_metadata.json \
                $MD_SAVE_ROOT/scannet_sim_metadata.json \
                $MD_SAVE_ROOT/replica_sim_metadata.json \
                $MD_SAVE_ROOT/hm3d_sim_metadata.json \
    --dataset-names "Gibson 4+" \
                    "Gibson" \
                    "RoboThor" \
                    "MP3D" \
                    "ScanNet" \
                    "Replica" \
                    "HM3D" \
    --frac-thresh 0.05 \
    --save-dir "results/reconstruction_completness"