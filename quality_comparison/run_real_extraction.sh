#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

REAL_ROOT="$SAVE_DIR_PATH/real_images"

python extract_gibson_real.py --dataset-dir $GIBSON_PANO_ROOT --save-dir $REAL_ROOT/gibson_real
python extract_mp3d_real.py --dataset-dir $MP3D_PANO_ROOT --save-dir $REAL_ROOT/mp3d_real