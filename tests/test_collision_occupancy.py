import unittest

import numpy as np
import trimesh

from src.datatypes.map import GRID_2D_FREE, GRID_2D_OCCUPIED
from src.planners.map_converter import collision_scene_to_occupancy_grid
from src.raycasting.scene import ObjectMesh, SceneModel, STATIC, face_normals


def _box_model(center, extents) -> SceneModel:
    """One collision-box instance whose transform exercises world conversion."""
    mesh = trimesh.creation.box(extents=extents)
    triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.asarray(center, dtype=np.float32)
    obj = ObjectMesh(
        local_verts=triangles,
        face_normal=face_normals(triangles),
        mesh_key="test-box",
    )
    return SceneModel(
        objects=[obj],
        transforms=np.asarray([transform]),
        motion_type=np.asarray([STATIC], dtype=np.int8),
        object_ids=np.asarray([1], dtype=np.int32),
        semantic_ids=np.asarray([0], dtype=np.int32),
        geometry="collision",
    )


class TestCollisionOccupancy(unittest.TestCase):
    def _grid(self, model):
        return collision_scene_to_occupancy_grid(
            model,
            resolution=0.1,
            bounds=(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([3.0, 1.0, 3.0], dtype=np.float32),
            ),
            floor_y=0.0,
            robot_height=0.5,
            robot_radius=0.2,
            floor_mask=np.ones((30, 30), dtype=bool),
        )

    def _cell(self, grid, x, z):
        col = int((x - grid.origin.position[0]) / grid.resolution)
        row = grid.height - 1 - int((z - grid.origin.position[2]) / grid.resolution)
        return grid.data[row, col]

    def test_tall_collision_box_blocks_footprint(self):
        grid = self._grid(_box_model(center=[1.5, 0.25, 1.5], extents=[0.2, 0.5, 0.2]))

        # The object itself and a point inside the circular robot-radius dilation
        # must be non-traversable; a distant supported cell stays free.
        self.assertEqual(self._cell(grid, 1.5, 1.5), GRID_2D_OCCUPIED)
        self.assertEqual(self._cell(grid, 1.8, 1.5), GRID_2D_OCCUPIED)
        self.assertEqual(self._cell(grid, 0.5, 0.5), GRID_2D_FREE)

    def test_geometry_above_robot_height_does_not_block_floor_cell(self):
        grid = self._grid(_box_model(center=[1.5, 0.9, 1.5], extents=[0.4, 0.2, 0.4]))

        self.assertEqual(self._cell(grid, 1.5, 1.5), GRID_2D_FREE)


if __name__ == "__main__":
    unittest.main()
