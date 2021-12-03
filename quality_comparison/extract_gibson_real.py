#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import tqdm
import imageio
import argparse
import numpy as np
import multiprocessing as mp

from py360convert import e2p


FOV = 90.0
HEIGHT = 300
WIDTH = 300
NUM_IMAGES_PER_PANO = 3


def extract_rgb_images(pano_path: str, save_prefix: str) -> None:
    pano = imageio.imread(pano_path)
    # Convert pano to images
    list_of_images = []
    locs = np.linspace(0, 1.0, num=10)[:-1]
    np.random.shuffle(locs)
    for loc in locs[:NUM_IMAGES_PER_PANO]:
        pimg = e2p(pano, (FOV, FOV), (loc - 0.5) * 360.0, 0.0, (HEIGHT, WIDTH))
        list_of_images.append(pimg)
    # Save images
    for idx, image in enumerate(list_of_images):
        path = save_prefix + f'_img_{idx:03d}.jpg'
        imageio.imwrite(path, image)


def _aux_fn(args):
    extract_rgb_images(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    pano_paths = sorted(glob.glob(os.path.join(args.dataset_dir, '*/pano/rgb/*_rgb.png')))
    print(f'Number of panoramic images: {len(pano_paths)}')

    inputs = [
        (pano_path, os.path.join(args.save_dir, f'pano_{i:06d}'))
        for i, pano_path in enumerate(pano_paths)
    ]

    os.makedirs(args.save_dir, exist_ok=True)
    pool = mp.Pool(args.num_workers, maxtasksperchild=2)

    _ = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(inputs)))