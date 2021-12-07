#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import os.path as osp
import random
from typing import Any, Dict, List, Optional, Tuple

import habitat_sim
import imageio
import numpy as np
import tqdm
from sklearn.cluster import DBSCAN

from common.utils import (
    convert_heading_to_quaternion,
    get_filtered_scenes,
    get_topdown_map,
)


def make_habitat_configuration(
    scene_path: str,
    hfov: int = 90,
    resolution: Tuple[int] = (300, 300),
    stage_json_path: Optional[str] = None,
):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    if stage_json_path is not None:
        backend_cfg.scene_dataset_config_file = stage_json_path
        backend_cfg.scene_id = "habitat/" + scene_path.split("/")[-1]
    else:
        backend_cfg.scene_id = scene_path

    # agent configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.uuid = "rgba"
    rgb_sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_cfg.resolution = resolution
    rgb_sensor_cfg.hfov = hfov
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = "depth"
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.resolution = resolution
    depth_sensor_cfg.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(scene_path: str, **kwargs: Any) -> habitat_sim.Simulator:
    sim_cfg = make_habitat_configuration(scene_path, **kwargs)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_floor_heights(
    sim: habitat_sim.Simulator, max_points_to_sample: int = 20000
) -> List[Dict[str, float]]:
    """Get heights of different floors in a scene. This is done in two steps.
    (1) Randomly samples navigable points in the scene.
    (2) Cluster the points based on discretized y coordinates to get floors.

    Args:
        sim: habitat simulator instance
        max_points_to_sample: number of navigable points to randomly sample
    """
    nav_points = []
    for _ in range(max_points_to_sample):
        nav_points.append(sim.pathfinder.get_random_navigable_point())
    nav_points = np.stack(nav_points, axis=0)
    y_coors = np.around(nav_points[:, 1], decimals=1)
    # cluster Y coordinates
    clustering = DBSCAN(eps=0.2, min_samples=2000).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # get floor extents in Y
    # each cluster corresponds to points from 1 floor
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
    floor_extents = sorted(floor_extents, key=lambda x: x["mean"])

    # reject floors that have too few points
    max_points = 0
    for fext in floor_extents:
        top_down_map = get_topdown_map(sim.pathfinder, fext["mean"])
        max_points = max(np.count_nonzero(top_down_map), max_points)
    clean_floor_extents = []
    for fext in floor_extents:
        top_down_map = get_topdown_map(sim.pathfinder, fext["mean"])
        num_points = np.count_nonzero(top_down_map)
        if num_points < 0.2 * max_points:
            continue
        clean_floor_extents.append(fext)

    return clean_floor_extents


def get_navmesh_extents_at_y(
    sim: habitat_sim.Simulator, y_bounds: Optional[Tuple[float]] = None
) -> Tuple[float]:
    if y_bounds is None:
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
    else:
        assert len(y_bounds) == 2
        assert y_bounds[0] < y_bounds[1]
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_vertices = navmesh_vertices[
            (y_bounds[0] <= navmesh_vertices[:, 1])
            & (navmesh_vertices[:, 1] <= y_bounds[1])
        ]
        lower_bound = navmesh_vertices.min(axis=0)
        upper_bound = navmesh_vertices.max(axis=0)
    return (lower_bound, upper_bound)


def get_dense_navmesh_vertices(
    sim: habitat_sim.Simulator, sampling_resolution: float = 0.5
) -> np.ndarray:

    navmesh_vertices = []
    floor_extents = get_floor_heights(sim)
    for fext in floor_extents:
        l_bound, u_bound = get_navmesh_extents_at_y(
            sim, y_bounds=(fext["min"] - 0.5, fext["max"] + 0.5)
        )
        x_range = np.arange(l_bound[0].item(), u_bound[0].item(), sampling_resolution)
        y = fext["mean"]
        z_range = np.arange(l_bound[2].item(), u_bound[2].item(), sampling_resolution)
        for x, z in itertools.product(x_range, z_range):
            if sim.pathfinder.is_navigable(np.array([x, y, z])):
                navmesh_vertices.append((np.array([x, y, z])))
    if len(navmesh_vertices) > 0:
        navmesh_vertices = np.stack(navmesh_vertices, axis=0)
    else:
        navmesh_vertices = np.zeros((0, 3))
    return navmesh_vertices


def get_scene_name(scene_path, dataset):
    if dataset == "replica":
        scene_name = scene_path.split("/")[-2].split(".")[0]
    else:
        scene_name = scene_path.split("/")[-1].split(".")[0]
    return scene_name


def extract_images_in_uniform_grid(
    scene_path: str,
    rgb_save_prefix: str,
    depth_save_prefix: str,
    hfov: int,
    resolution: List[int],
    sampling_resolution: float = 0.5,
    num_rotations: int = 4,
    sim_kwargs: Dict[Any, Any] = None,
) -> Tuple[List[str]]:
    if sim_kwargs is None:
        sim_kwargs = {}
    sim = robust_load_sim(scene_path, hfov=hfov, resolution=resolution, **sim_kwargs)
    agent = sim.get_agent(0)
    rgb_paths = []
    depth_paths = []
    if not sim.pathfinder.is_loaded:
        sim.close()
        return rgb_paths, depth_paths

    # Get dense navmesh vertices
    navmesh_vertices = get_dense_navmesh_vertices(
        sim, sampling_resolution=sampling_resolution
    )

    count = 0
    for idx in range(len(navmesh_vertices)):
        loc = navmesh_vertices[idx]
        for heading in np.linspace(0, 360, num_rotations + 1)[:-1]:
            rot = convert_heading_to_quaternion(heading)
            agent_state = agent.get_state()
            agent_state.position = loc
            agent_state.rotation = rot
            agent.set_state(agent_state, reset_sensors=True)
            obs = sim.get_sensor_observations()
            rgb = obs["rgba"][..., :3]
            depth = obs["depth"]
            rgb_path = rgb_save_prefix + f"_img_{count:05d}.jpg"
            depth_path = depth_save_prefix + f"_img_{count:05d}.npy"
            imageio.imwrite(rgb_path, rgb)
            np.savez_compressed(depth_path, depth=depth)
            rgb_paths.append(rgb_path)
            depth_paths.append(depth_path)
            count += 1
    sim.close()
    return rgb_paths, depth_paths


def _aux_fn(args):
    return extract_images_in_uniform_grid(*args)


HFOV = 90
RESOLUTION = [300, 300]
NUM_ROTATIONS = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--filter-scenes", type=str, default="")
    parser.add_argument("--rgb-save-dir", type=str, default="")
    parser.add_argument("--depth-save-dir", type=str, default="")
    parser.add_argument("--json-save-path", type=str, default="data.json")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["gibson", "hm3d", "mp3d", "replica", "scannet", "robothor"],
    )
    parser.add_argument("--stage-json-path", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--sampling-resolution", type=float, default=1.0)

    args = parser.parse_args()

    random.seed(123)
    np.random.seed(123)

    if args.dataset_name in ["gibson", "mp3d", "scannet", "robothor"]:
        scenes = glob.glob(f"{args.dataset_dir}/**/*.glb", recursive=True)
    elif args.dataset_name == "hm3d":
        scenes = []
        for split in ["train", "val"]:
            scenes += glob.glob(f"{args.dataset_dir}/{split}/**/*.glb", recursive=True)
    elif args.dataset_name in ["replica"]:
        scenes = glob.glob(f"{args.dataset_dir}/*/mesh.ply", recursive=True)

    if args.filter_scenes != "":
        scenes = get_filtered_scenes(scenes, args.filter_scenes)
    scenes = sorted(scenes)
    # Filter out basis scenes
    scenes = [s for s in scenes if ".basis." not in s]

    print(f"Number of scenes in {args.dataset_dir}: {len(scenes)}")

    sim_kwargs = {"stage_json_path": args.stage_json_path}
    inputs = []
    for scene_path in scenes:
        suffix = get_scene_name(scene_path, args.dataset_name)
        inputs.append(
            (
                scene_path,
                osp.join(args.rgb_save_dir, suffix),
                osp.join(args.depth_save_dir, suffix),
                HFOV,
                RESOLUTION,
                args.sampling_resolution,
                NUM_ROTATIONS,
                sim_kwargs,
            )
        )

    os.makedirs(args.rgb_save_dir, exist_ok=True)
    os.makedirs(args.depth_save_dir, exist_ok=True)

    context = mp.get_context("forkserver")
    pool = context.Pool(processes=args.num_processes, maxtasksperchild=2)

    all_paths = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(inputs)))

    # Create metadata
    metadata = []
    for scene_idx, scene_paths in enumerate(all_paths):
        scene_name = get_scene_name(scenes[scene_idx], args.dataset_name)
        rgb_paths, depth_paths = scene_paths
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            md = {"rgb_path": rgb_path, "depth_path": depth_path}
            metadata.append(md)

    with open(args.json_save_path, "w") as fp:
        json.dump(metadata, fp)
