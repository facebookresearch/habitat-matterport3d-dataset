import glob
import tqdm
import trimesh
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

from typing import Any, Dict, List, Callable
from utils import robust_load_sim, get_filtered_scenes
from metrics import (
    compute_navigable_area,
    compute_navigation_complexity,
    compute_scene_clutter,
    compute_floor_area,
)


VALID_METRICS: List[str] = [
    "navigable_area",
    "navigation_complexity",
    "scene_clutter",
    "floor_area",
]


METRIC_TO_FN_MAP: Dict[str, Callable] = {
    "navigable_area": compute_navigable_area,
    "navigation_complexity": compute_navigation_complexity,
    "scene_clutter": compute_scene_clutter,
    "floor_area": compute_floor_area,
}


METRICS_TO_AVERAGE: List[str] = ["navigation_complexity", "scene_clutter"]


def compute_metrics(
    scene_path: str,
    voxel_size: float,
    metrics: List[str] = VALID_METRICS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Computes the 3D scene metrics for a given file.

    Args:
        scene_path: path to the scene file (glb / ply)
        voxel_size: specifies the voxel size for scene simplification
        metrics: list of metrics to compute

    Outputs:
        metric_values: a dict mapping from the required metric names to values
    """
    # sanity check
    for metric in metrics:
        assert metric in VALID_METRICS
    # load scene in habitat_simulator and trimesh
    hsim = robust_load_sim(scene_path)
    trimesh_scene = trimesh.load(scene_path)
    # Simplify scene-mesh for faster metric computation
    # Does not impact the final metrics much
    o3d_scene = o3d.geometry.TriangleMesh()
    vertices = np.array(trimesh_scene.triangles).reshape(-1, 3)
    faces = np.arange(0, len(vertices)).reshape(-1, 3)
    o3d_scene.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_scene.triangles = o3d.utility.Vector3iVector(faces)
    o3d_scene = o3d_scene.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    if verbose:
        print(
            f"=====> Downsampled mesh from {len(trimesh_scene.triangles)} "
            f"to {len(o3d_scene.triangles)}"
        )
    trimesh_scene = trimesh.Trimesh()
    trimesh_scene.vertices = np.array(o3d_scene.vertices)
    trimesh_scene.faces = np.array(o3d_scene.triangles)

    metric_values = {}
    for metric in metrics:
        metric_values[metric] = METRIC_TO_FN_MAP[metric](hsim, trimesh_scene)
    metric_values["scene"] = scene_path
    hsim.close()
    return metric_values


def _aux_fn(inputs: Any) -> Any:
    return compute_metrics(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+", default=VALID_METRICS)
    parser.add_argument("--filter-scenes", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--scan-pattern", type=str, default="*.glb")
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--n-processes", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    scenes = glob.glob(f"{args.dataset_root}/**/{args.scan_pattern}", recursive=True)
    if args.filter_scenes != "":
        scenes = get_filtered_scenes(scenes, args.filter_scenes)
    scenes = sorted(scenes)
    # Filter out basis scenes
    scenes = [s for s in scenes if ".basis." not in s]

    if args.verbose:
        print(f"Number of scenes in {args.dataset_root}: {len(scenes)}")

    context = mp.get_context("forkserver")
    pool = context.Pool(processes=args.n_processes, maxtasksperchild=2)
    inputs = [[scene, args.voxel_size, args.metrics, args.verbose] for scene in scenes]

    stats = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(scenes)))
    stats = pd.DataFrame(stats)
    stats.set_index("scene", inplace=True)
    print("============= Metrics =============")
    print(f"Number of scenes: {len(scenes)}")
    for metric in args.metrics:
        if metric in METRICS_TO_AVERAGE:
            v = stats[metric].to_numpy().mean().item()
        else:
            v = stats[metric].to_numpy().sum().item()
        print(f"{metric:<30s} | {v:.3f}")

    if args.save_path != "":
        stats.to_csv(args.save_path)
