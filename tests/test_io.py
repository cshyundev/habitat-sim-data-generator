import os
import unittest
import numpy as np
import tempfile

from src.datatypes.pose import Pose3D
from src.utils.io import save_poses_to_csv, load_poses_from_csv

class TestPoseIO(unittest.TestCase):
    def setUp(self):
        # Generate some sample Pose3D objects
        self.poses = [
            Pose3D(
                position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            ),
            Pose3D(
                position=np.array([-5.4, 0.25, 12.87], dtype=np.float32),
                orientation=np.array([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float32)
            ),
            Pose3D(
                position=np.array([0.0, -1.5, -3.2], dtype=np.float32),
                orientation=np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float32)
            )
        ]
        
    def test_save_and_load_poses(self):
        """Verify that poses can be exported to CSV and loaded back with high numerical precision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_poses.csv")
            
            # Save
            save_poses_to_csv(self.poses, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            # Load
            loaded_poses = load_poses_from_csv(csv_path)
            self.assertEqual(len(loaded_poses), len(self.poses))
            
            for original, loaded in zip(self.poses, loaded_poses):
                # Verify positions are close
                np.testing.assert_allclose(loaded.position, original.position, rtol=1e-5, atol=1e-5)
                # Verify orientations are close
                np.testing.assert_allclose(loaded.orientation, original.orientation, rtol=1e-5, atol=1e-5)
                # Verify yaw calculation matches
                self.assertAlmostEqual(loaded.yaw, original.yaw, places=5)

if __name__ == "__main__":
    unittest.main()
