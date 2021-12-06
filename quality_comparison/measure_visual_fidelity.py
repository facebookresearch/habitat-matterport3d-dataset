#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch_fidelity


def measure_visual_fidelity(args):
    metrics = torch_fidelity.calculate_metrics(
        input1=args.sim_path,
        input2=args.real_path,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
    )
    for k, v in metrics.items():
        print(f"{k:<40s}: {v}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--sim-path", type=str, required=True)

    args = parser.parse_args()

    measure_visual_fidelity(args)
