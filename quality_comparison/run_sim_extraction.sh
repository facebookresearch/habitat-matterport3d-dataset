#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SAVE_ROOT="$SAVE_DIR_PATH/simulated_images"
MD_SAVE_ROOT="$SAVE_DIR_PATH/simulated_images/rgb2depth_metadata"

mkdir "$SAVE_ROOT"
mkdir "$MD_SAVE_ROOT"

python extract_sim.py \
   --dataset-dir "$GIBSON_ROOT" \
   --filter-scenes data/gibson_4_plus.txt \
   --rgb-save-dir "$SAVE_ROOT"/gibson_4_plus_sim \
   --depth-save-dir "$SAVE_ROOT"/gibson_4_plus_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/gibson_4_plus_sim_metadata.json \
   --dataset-name gibson

python extract_sim.py \
   --dataset-dir "$GIBSON_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/gibson_sim \
   --depth-save-dir "$SAVE_ROOT"/gibson_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/gibson_sim_metadata.json \
   --dataset-name gibson

python extract_sim.py \
   --dataset-dir "$REPLICA_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/replica_sim \
   --depth-save-dir "$SAVE_ROOT"/replica_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/replica_sim_metadata.json \
   --dataset-name replica \
   --sampling-resolution 0.25

python extract_sim.py \
   --dataset-dir "$MP3D_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/mp3d_sim \
   --depth-save-dir "$SAVE_ROOT"/mp3d_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/mp3d_sim_metadata.json \
   --dataset-name mp3d

python extract_sim.py \
   --dataset-dir "$HM3D_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/hm3d_sim \
   --depth-save-dir "$SAVE_ROOT"/hm3d_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/hm3d_sim_metadata.json \
   --dataset-name hm3d

python extract_sim.py \
   --dataset-dir "$SCANNET_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/scannet_sim \
   --depth-save-dir "$SAVE_ROOT"/scannet_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/scannet_sim_metadata.json \
   --dataset-name scannet

python extract_sim.py \
   --dataset-dir "$ROBOTHOR_ROOT" \
   --rgb-save-dir "$SAVE_ROOT"/robothor_sim \
   --depth-save-dir "$SAVE_ROOT"/robothor_sim_depth \
   --json-save-path "$MD_SAVE_ROOT"/robothor_sim_metadata.json \
   --dataset-name robothor \
   --stage-json-path "data/robothor.scene_dataset_config.json"
