import habitat_sim


def make_habitat_configuration(scene_path, use_sensor=False):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    # agent configuration
    sensor_cfg = habitat_sim.SensorSpec()
    sensor_cfg.resolution = [1080, 960]
    sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg] if use_sensor else []

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(scene_path):
    sim_cfg = make_habitat_configuration(scene_path, use_sensor=False)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        hsim.close()
        sim_cfg = make_habitat_configuration(scene_path, use_sensor=True)
        hsim = habitat_sim.Simulator(sim_cfg)
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_filtered_scenes(scenes, filter_scenes_path):
    with open(filter_scenes_path, "r") as fp:
        filter_scenes = fp.readlines()
    filter_scenes = [f.strip("\n") for f in filter_scenes]
    filtered_scenes = []
    for scene in scenes:
        scene_name = scene.split("/")[-1][: -len(".glb")]
        if scene_name in filter_scenes:
            filtered_scenes.append(scene)
    return filtered_scenes
