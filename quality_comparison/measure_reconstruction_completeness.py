import os
import re
import tqdm
import json
import imageio
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import os.path as osp
import multiprocessing as mp

from collections import defaultdict

sns.set(
    style='whitegrid',
    rc={"lines.linewidth": 2.5, "ytick.left": True, "xtick.bottom": True}
)
sns.set_palette("bright")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)

mpl.rc('axes', edgecolor='lightgray')


# ========= Font properties for the plots ==============
legend_font = font_manager.FontProperties(
    style='normal',
    size=20
)

axes_font = {
    'style'  : 'normal',
    'size'   : 20
}

ticks_font = font_manager.FontProperties(
    style='normal',
    size=20
)

bright_palette = sns.color_palette("bright")
pastel_palette = sns.color_palette("pastel")
muted_palette = sns.color_palette("muted")
COLOR_MAPPING = {
    "Gibson 4+": pastel_palette[0],
    "Gibson": bright_palette[0],
    "RoboThor": bright_palette[6],
    "MP3D": bright_palette[2],
    "ScanNet": bright_palette[1],
    "Replica": bright_palette[4],
    "HM3D": bright_palette[3],
}


def is_image_defective(info):
    mode = info["mode"]
    depth, rgb = None, None
    use_depth = (mode in ["depth", "rgb+depth"])
    use_rgb = (mode in ["rgb", "rgb+depth"])
    # Read images
    if use_depth:
        depth = np.load(info['depth_path'] + '.npz')['depth']
    if use_rgb:
        rgb = imageio.imread(info['rgb_path'])
    # Get fraction of defective values
    if use_depth and use_rgb:
        mask = (depth == 0) | np.all(rgb == 0, axis=2)
    elif use_depth:
        mask = (depth == 0)
    else:
        mask = np.all(rgb == 0, axis=2)
    frac = float(np.count_nonzero(mask)) / mask.size
    # Get scene name
    match = re.match('(.*)_img_(.*).npy', osp.basename(info['depth_path']))
    assert match is not None
    scene_name = match.group(1)
    has_defect = 1.0 if frac > args.frac_thresh else 0.0
    return scene_name, has_defect, frac


def measure_reconstruction_completeness(args):
    pool = mp.Pool(args.num_workers)

    stats = []
    per_image_stats = []
    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = f'{args.save_dir}/dataset_stats.csv'
    per_image_stats_path = f'{args.save_dir}/dataset_image_stats.csv'
    if not osp.isfile(stats_path):
        for dataset_name, json_path in zip(args.dataset_names, args.json_paths):
            print(f"=======> Evaluating {dataset_name}")
            dataset_info = json.load(open(json_path))
            # Update dataset_info to include mode
            assert "mode" not in dataset_info[0], \
                "dataset_info already contains the key 'mode'"
            dataset_info = [{"mode": args.mode, **di} for di in dataset_info]
            # Compute stats over the complete dataset
            dataset_stats = list(tqdm.tqdm(
                pool.imap(is_image_defective, dataset_info),
                total=len(dataset_info)
            ))
            # Compute scene-specific stats
            scene_stats = defaultdict(list)
            for info, (scene_name, has_defect, frac_defect) in zip(dataset_info, dataset_stats):
                scene_stats[scene_name].append(
                    (has_defect, frac_defect, info['rgb_path'], info['depth_path'])
                )
            for scene_name, defects_info in scene_stats.items():
                defects = [di[0] for di in defects_info]
                stats.append({
                    "scene": scene_name,
                    "% defects": np.mean(defects).item() * 100.0,
                    "dataset": dataset_name
                })
                per_image_stats += [
                    {"scene": scene_name,
                     "has defect": di[0],
                     "frac defects": di[1],
                     "rgb_path": di[2],
                     "depth_path": di[3]} for di in defects_info
                ]
        stats = pd.DataFrame(stats)
        stats.to_csv(stats_path, index=False)
        per_image_stats = pd.DataFrame(per_image_stats)
        per_image_stats.to_csv(per_image_stats_path, index=False)
    else:
        stats = pd.read_csv(stats_path, index_col=False)
        per_image_stats = pd.read_csv(per_image_stats_path, index_col=False)

    plt.figure(figsize=args.figsize)
    g = sns.histplot(
        stats, x="% defects", element='step',
        hue="dataset", fill=False, bins=60,
        palette=[COLOR_MAPPING[d] for d in args.dataset_names],
    )
    for label in plt.xticks()[1] + plt.yticks()[1]:
        label.set_fontproperties(ticks_font)
    plt.xlabel("% defects", fontdict=axes_font)
    plt.ylabel("# scenes", fontdict=axes_font)
    ax = plt.gca()
    ax.get_legend().remove()
    plt.xlim(0, 100)
    plt.ylim(1, 1000)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(LogLocator())
    plt.tight_layout()

    plt.savefig(f'{args.save_dir}/histplog.png')

    # Print average stats per dataset
    grouped_stats = stats.set_index('% defects', drop=True).groupby('dataset').groups
    for k, v in grouped_stats.items():
        v_mean = np.mean(v)
        print(f'{k}: {v_mean:.4f}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-paths', type=str, required=True, nargs="+")
    parser.add_argument('--dataset-names', type=str, required=True, nargs="+")
    parser.add_argument('--frac-thresh', type=float, default=0.05,
                        help="threshold for deciding presence of mesh defects")
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='./')
    parser.add_argument('--mode', type=str, default="rgb+depth",
                        choices=["depth", "rgb", "rgb+depth"])
    parser.add_argument("--figsize", type=int, default=(8, 5), nargs="+")

    args = parser.parse_args()

    assert len(args.json_paths) == len(args.dataset_names)

    measure_reconstruction_completeness(args)