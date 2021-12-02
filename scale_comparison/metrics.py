import math
import scipy
import trimesh
import numpy as np
import habitat_sim

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from typing import Any, Dict, List


EPS = 1e-10


def get_geodesic_distance(
    hsim: habitat_sim.Simulator, p1: np.ndarray, p2: np.ndarray
) -> float:
    """Computes the geodesic distance between two points."""
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    hsim.pathfinder.find_path(path)
    return path.geodesic_distance


def get_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes the euclidean distance between two points."""
    return np.linalg.norm(p1 - p2).item()


def get_navcomplexity(
    hsim: habitat_sim.Simulator, p1: np.ndarray, p2: np.ndarray
) -> float:
    """Computes the navigation complexity between two points in a scene."""
    geod = get_geodesic_distance(hsim, p1, p2)
    eucd = get_euclidean_distance(p1, p2)
    return geod / (eucd + EPS)


def get_triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """
    Measure the area of mesh triangles.
    Args:
        triangles: (N, 3, 3) ndarray with dimension 1 representing 3 vertices
    """
    vtr10 = triangles[:, 1] - triangles[:, 0]  # (N, 3)
    vtr20 = triangles[:, 2] - triangles[:, 0]  # (N, 3)
    area = 0.5 * np.linalg.norm(np.abs(np.cross(vtr10, vtr20, axis=1)), axis=1)
    return area


def transform_coordinates_hsim_to_trimesh(xyz: np.ndarray) -> np.ndarray:
    """
    Transforms points from hsim coordinates to trimesh.

    Habitat conventions: X is rightward, Y is upward, -Z is forward
    Trimesh conventions: X is rightward, Y is forward, Z is upward

    Args:
        xyz: (N, 3) array of coordinates
    """
    xyz_trimesh = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    return xyz_trimesh


def get_floor_navigable_extents(
    hsim: habitat_sim.Simulator, num_points_to_sample: int = 20000
) -> List[Dict[str, float]]:
    """
    Function to estimate the number of floors in a 3D scene and the Y extents
    of the navigable space on each floor. It samples a random number
    of navigable points and clusters them based on their height coordinate.
    Each cluster corresponds to a floor, and the points within a cluster
    determine the extents of the navigable surfaces in a floor.
    """
    # randomly sample navigable points
    random_navigable_points = []
    for i in range(num_points_to_sample):
        point = hsim.pathfinder.get_random_navigable_point()
        if np.isnan(point).any() or np.isinf(point).any():
            continue
        random_navigable_points.append(point)
    random_navigable_points = np.array(random_navigable_points)
    # cluster the rounded y_coordinates using DBScan
    y_coors = np.around(random_navigable_points[:, 1], decimals=1)
    clustering = DBSCAN(eps=0.2, min_samples=500).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # estimate floor extents
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})

    return floor_extents


def compute_navigable_area(
    hsim: habitat_sim.Simulator, *args: Any, **kwargs: Any
) -> float:
    """
    Navigable area (m^2) measures the total scene area that is actually
    navigable in the scene. This is computed for a cylindrical robot with radius
    0.1m and height 1.5m using the AI Habitat navigation mesh implementation.
    This excludes points that are not reachable by the robot. Higher values
    indicate larger quantity and diversity of viewpoints for a robot.
    """
    return hsim.pathfinder.navigable_area


def compute_navigation_complexity(
    hsim: habitat_sim.Simulator,
    *args: Any,
    max_pairs_to_sample: int = 20000,
    max_trials_per_pair: int = 10,
    **kwargs: Any,
) -> float:
    """
    Navigation complexity measures the difficulty of navigating in a scene.
    This is computed as the maximum ratio of geodesic path to euclidean
    distances between any two navigable locations in the scene. Higher values
    indicate more complex layouts with navigation paths that deviate
    significantly from straight-line paths.

    Args:
        hsim: habitat simulator instance
        max_pairs_to_sample: the maximum number of random point pairs to sample
        max_trials_per_pair: the maximum trials to find a paired point p2 for
            a given point p1
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    navcomplexity = 0.0
    num_sampled_pairs = 0
    while num_sampled_pairs < max_pairs_to_sample:
        num_sampled_pairs += 1
        p1 = hsim.pathfinder.get_random_navigable_point()
        num_trials = 0
        while num_trials < max_trials_per_pair:
            num_trials += 1
            p2 = hsim.pathfinder.get_random_navigable_point()
            # Different floors
            if abs(p1[1] - p2[1]) > 0.5:
                continue
            cur_navcomplexity = get_navcomplexity(hsim, p1, p2)
            # Ignore disconnected pairs
            if math.isinf(cur_navcomplexity):
                continue
            navcomplexity = max(navcomplexity, cur_navcomplexity)

    return navcomplexity


def compute_scene_clutter(
    hsim: habitat_sim.Simulator,
    trimesh_scene: trimesh.parent.Geometry,
    closeness_thresh: float = 0.5
) -> float:
    """
    Scene clutter measures amount of clutter in the scene. This is computed as
    the ratio between the raw scene mesh area within 0.5m of the navigable
    regions and the navigable space. We restrict to 0.5m to only pick the
    surfaces that are near navigable spaces in the building
    (e.g., furniture, and interior walls), and to ignore other surfaces outside
    the building. This is implemented in the same way as by Xia et al. to
    make the reported statistics comparable. Higher values are better and
    indicate more cluttered scenes that provide more obstacles for navigation.

    Args:
        hsim: habitat simulator instance
        trimesh_scene: 3D scene loaded in trimesh
        closeness_thresh: a distance threshold for points on the mesh to be
            considered "close" to navigable space.
    
    Reference:
        Xia, Fei, et al.
        "Gibson env: Real-world perception for embodied agents."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    mesh_triangles = np.copy(trimesh_scene.triangles)
    # convert habitat navmesh to a trimesh scene
    navmesh_vertices = np.array(hsim.pathfinder.build_navmesh_vertices())
    navmesh_vertices = transform_coordinates_hsim_to_trimesh(navmesh_vertices)
    ## three consecutive vertices form a triangle face
    navmesh_faces = np.arange(0, navmesh_vertices.shape[0], dtype=np.uint32)
    navmesh_faces = navmesh_faces.reshape(-1, 3)
    navmesh_triangles = navmesh_vertices.reshape(-1, 3, 3)
    navmesh_centroids = navmesh_triangles.mean(axis=1)
    navmesh = trimesh.Trimesh(vertices=navmesh_vertices, faces=navmesh_faces)
    # Find closest distance between a mesh_triangle and the navmesh
    # This is approximated by measuring the distance between each vertex and
    # centroid of a mesh_triangle to the navmesh surface
    ## (1) pre-filtering to remove unrelated mesh_triangles
    tree = scipy.spatial.cKDTree(navmesh_centroids)
    mesh_centroids = mesh_triangles.mean(axis=1)[:, np.newaxis, :]
    mindist, _ = tree.query(mesh_centroids)
    valid_mask = mindist[:, 0] <= 2 * closeness_thresh
    mesh_triangles = mesh_triangles[valid_mask]
    mesh_centroids = mesh_centroids[valid_mask]
    # (2) min distance b/w vertex / centroid of a mesh triangle to navmesh
    mesh_tricents = np.concatenate(
        [mesh_triangles, mesh_centroids], axis=1
    )  # (N, 4, 3)
    mesh_tricents = mesh_tricents.reshape(-1, 3)
    _, d2navmesh, _ = navmesh.nearest.on_surface(mesh_tricents)  # (N * 4, )
    d2navmesh = d2navmesh.reshape(-1, 4).min(axis=1)  # (N, )
    closest_mesh_triangles = mesh_triangles[(d2navmesh < closeness_thresh)]
    clutter_area = get_triangle_areas(closest_mesh_triangles).sum().item()
    navmesh_area = hsim.pathfinder.navigable_area
    clutter = clutter_area / (navmesh_area + EPS)

    return clutter


def compute_floor_area(
    hsim: habitat_sim.Simulator,
    trimesh_scene: trimesh.parent.Geometry,
    floor_limit: float = 0.5
) -> float:
    """
    Floor area (m^2) measures the overall extents of the floor regions in the
    scene. This is the area of the 2D convex hull of all navigable locations in
    a floor. For scenes with multiple floors, the floor space is summed over all
    floors. This is implemented in the same way as by Xia et al. to make the
    reported statistics comparable. Higher values indicate the presence of more
    navigation space and rooms.

    Args:
        hsim: habitat simulator instance
        trimesh_scene: 3D scene loaded in trimesh
        floor_limit: defines the maximum height above the navigable space
            that is considered as a part of the current floor.

    Reference:
        Xia, Fei, et al.
        "Gibson env: Real-world perception for embodied agents."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    floor_extents = get_floor_navigable_extents(hsim)
    mesh_vertices = trimesh_scene.triangles.reshape(-1, 3)
    # Z axis in trimesh is vertically upward
    floor_area = 0.0
    for fext in floor_extents:
        mask = (mesh_vertices[:, 2] >= fext["min"]) & (
            mesh_vertices[:, 2] < fext["max"] + floor_limit
        )
        floor_convex_hull = ConvexHull(mesh_vertices[mask, :2])
        # convex_hull.volume computes the area for 2D convex hull
        floor_area += floor_convex_hull.volume
    return floor_area