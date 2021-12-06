#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Turn off non-critical habitat-sim logging
export GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet

# Gibson 4+
python compute_scene_metrics.py --dataset-root "$GIBSON_ROOT" --filter-scenes data/gibson_4_plus.txt

# Gibson Full
python compute_scene_metrics.py --dataset-root "$GIBSON_ROOT"

# ROBOTHOR
python compute_scene_metrics.py --dataset-root "$ROBOTHOR_ROOT"

# Replica
python compute_scene_metrics.py --dataset-root "$REPLICA_ROOT" --scan-patterns "**/mesh_semantic.ply"

# ScanNet
python compute_scene_metrics.py --dataset-root "$SCANNET_ROOT"

# MP3D
python compute_scene_metrics.py --dataset-root "$MP3D_ROOT"

# HM3D
python compute_scene_metrics.py --dataset-root "$HM3D_ROOT" --scan-patterns "train/**/*.glb" "val/**/*.glb"
