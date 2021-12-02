#!/bin/bash

# Turn off non-critical habitat-sim logging
export GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet


GIBSON_ROOT=<PATH TO GIBSON glbs>
MP3D_ROOT=<PATH TO MP3D glbs>
ROBOTHOR_ROOT=<PATH TO ROBOTHOR glbs>
HABITAT_MATTERPORT_ROOT=<PATH TO HM3D glbs>
REPLICA_ROOT=<PATH TO REPLICA plys>
SCANNET_ROOT=<PATH TO SCANNET glbs>


# Gibson 4+
python compute_scene_metrics.py --dataset-root $GIBSON_ROOT --filter-scenes data/gibson_4_plus.txt

# Gibson 2+
python compute_scene_metrics.py --dataset-root $GIBSON_ROOT --filter-scenes data/gibson_2_plus.txt

# Gibson Full
python compute_scene_metrics.py --dataset-root $GIBSON_ROOT

# ROBOTHOR
python compute_scene_metrics.py --dataset-root $ROBOTHOR_ROOT

# Replica
python compute_scene_metrics.py --dataset-root $REPLICA_ROOT --scan-pattern "mesh_semantic.ply"

# ScanNet
python compute_scene_metrics.py --dataset-root $SCANNET_ROOT

# MP3D
python compute_scene_metrics.py --dataset-root $MP3D_ROOT

# HM3D
python compute_scene_metrics.py --dataset-root $HABITAT_MATTERPORT_ROOT