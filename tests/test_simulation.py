import numpy as np
import pybullet as p

from open_vocab_grasping.config import load_config
from open_vocab_grasping.perception.pointcloud import rgbd_to_pointcloud
from open_vocab_grasping.planning.ik import solve_ik
from open_vocab_grasping.planning.collision import joint_path_collision_free
from open_vocab_grasping.pipeline import resolve_scene_target
from open_vocab_grasping.simulation.camera import segmentation_body_ids
from open_vocab_grasping.simulation.world import SimulationWorld


def config():
    return load_config("configs/cpu_smoke.yaml")


def test_seeded_scene_is_reproducible() -> None:
    with SimulationWorld.create(config(), 42) as first:
        positions_a = {name: first.object_position(name) for name in first.objects}
    with SimulationWorld.create(config(), 42) as second:
        positions_b = {name: second.object_position(name) for name in second.objects}
    for name in positions_a:
        np.testing.assert_allclose(positions_a[name], positions_b[name], atol=1e-10)


def test_table_and_object_pointcloud_match_simulation() -> None:
    with SimulationWorld.create(config(), 5) as world:
        obs = world.camera.capture()
        points, _ = rgbd_to_pointcloud(obs.rgb, obs.depth_m, obs.intrinsic, obs.T_world_camera)
        body_ids = segmentation_body_ids(obs.segmentation).ravel()
        table_points = points[body_ids == world.table_id]
        assert len(table_points) > 100
        assert abs(float(np.median(table_points[:, 2]))) < 0.006
        for name, item in world.objects.items():
            object_points = points[body_ids == item.body_id]
            assert len(object_points) > 5
            assert np.linalg.norm(np.median(object_points, axis=0) - world.object_position(name)) < 0.07


def test_ik_rejects_far_pose() -> None:
    with SimulationWorld.create(config(), 1) as world:
        rotation = np.diag([1.0, -1.0, -1.0])
        result = solve_ik(world.robot, np.array([4.0, 0.0, 4.0]), rotation)
        assert not result.reachable


def test_compound_target_requires_matching_color() -> None:
    with SimulationWorld.create(config(), 1) as world:
        assert resolve_scene_target(world, "red bottle") == "bottle"
        try:
            resolve_scene_target(world, "red bowl")
        except ValueError:
            pass
        else:
            raise AssertionError("A mismatched color prompt must not map to the bowl truth object")


def test_joint_path_collision_detects_external_obstacle() -> None:
    with SimulationWorld.create(config(), 2) as world:
        position, _ = world.robot.end_effector_pose()
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], physicsClientId=world.client_id
        )
        obstacle = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=position.tolist(),
            physicsClientId=world.client_id,
        )
        free, reason = joint_path_collision_free(
            world.robot, world.robot.home[None, :], [obstacle], check_self_collision=False
        )
        assert not free
        assert reason == f"collision_body_{obstacle}"
