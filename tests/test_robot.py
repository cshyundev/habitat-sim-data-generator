import unittest

import numpy as np
import habitat_sim

from src.robot import DEFAULT_HEIGHT, DEFAULT_URDF, add_robot, cylinder_urdf


def _make_empty_sim():
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = "NONE"  # empty stage
    cfg.enable_physics = True
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))


def _link_translations(ao):
    """name -> link scene-node translation (Habitat frame), as a dict."""
    out = {}
    for lid in ao.get_link_ids():
        node = ao.get_link_scene_node(lid)
        t = node.translation
        out[ao.get_link_name(lid)] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return out


@unittest.skipUnless(
    habitat_sim.built_with_bullet, "URDF articulated objects require a Bullet build"
)
class TestDefaultRobot(unittest.TestCase):
    def setUp(self):
        self.sim = _make_empty_sim()

    def tearDown(self):
        self.sim.close()

    def test_default_file_contains_sensor_frames(self):
        """The shipped URDF owns the default sensor mount frames."""
        from_file = add_robot(self.sim, urdf=DEFAULT_URDF)
        file_links = _link_translations(from_file)
        self.assertIn("lidar_link", file_links)
        self.assertIn("camera_link", file_links)
        self.assertIn("imu_link", file_links)

    def test_lidar_sits_on_top(self):
        ao = add_robot(self.sim)
        links = _link_translations(ao)
        self.assertIn("lidar_link", links)
        # habitat-sim keeps the URDF (Z-up) frame: the top-mounted LiDAR is at z == height.
        self.assertAlmostEqual(links["lidar_link"][2], DEFAULT_HEIGHT, places=5)

    def test_custom_dimensions(self):
        ao = add_robot(self.sim, height=0.8, radius=0.3)
        links = _link_translations(ao)
        self.assertAlmostEqual(links["lidar_link"][2], 0.8, places=5)


class TestCylinderUrdfText(unittest.TestCase):
    def test_asset_matches_generator_defaults(self):
        """The shipped asset keeps the generator's default body and lidar frame."""
        with open(DEFAULT_URDF) as f:
            on_disk = f.read()
        generated = cylinder_urdf()
        # Compare structurally (ignore leading comment/whitespace differences).
        self.assertIn('<cylinder radius="0.25" length="0.5"/>', on_disk)
        self.assertIn('<cylinder radius="0.25" length="0.5"/>', generated)
        self.assertIn('<origin xyz="0 0 0.5" rpy="0 0 0"/>', on_disk)  # lidar at top
        self.assertIn('<origin xyz="0 0 0.5" rpy="0 0 0"/>', generated)


if __name__ == "__main__":
    unittest.main()
