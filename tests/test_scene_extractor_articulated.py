"""Articulated-object link extraction against the real habitat scene-node API,
plus the extractor's fail-loud contract.

habitat-sim exposes ``SceneNode.absolute_transformation_matrix`` as a *method*
(returning a magnum ``Matrix4``), not a property. Treating it as a property
made ``np.asarray(<bound method>)`` raise inside the per-link try/except, so
every articulated link (fridge, kitchen counter, cupboards, doors...) was
silently dropped from the SceneModel -- missing from scene markers AND from
GPU ray casting. These tests drive the extractor with a fake sim whose node
mimics the real method-style API.

The fail-loud tests pin the opposite side of that lesson: extraction inputs
that used to be swallowed (stage template failure, unloadable instance mesh,
missing/empty articulated URDF visuals) must raise, not shrink the scene.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest

import numpy as np
import trimesh

from src.raycasting.scene_extractor import (
    extract_scene_model,
    read_dynamic_transforms,
)


def _write_quad_obj(path: str) -> None:
    """Unit quad in the local XY plane (normal +z)."""
    verts = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def _write_floor_obj(path: str) -> None:
    """Stage stand-in: quad in the XZ plane at y=-5, far from any test ray."""
    verts = np.array(
        [[-9.0, -5.0, -9.0], [9.0, -5.0, -9.0], [9.0, -5.0, 9.0], [-9.0, -5.0, 9.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def _write_urdf(path: str, mesh_filename: str) -> None:
    with open(path, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n'
            '<robot name="cabinet">\n'
            '  <link name="door">\n'
            "    <visual>\n"
            f'      <geometry><mesh filename="{mesh_filename}"/></geometry>\n'
            "    </visual>\n"
            "  </link>\n"
            "</robot>\n"
        )


class _MethodTransformNode:
    """Mimics habitat's SceneNode: the world matrix is a METHOD, not a property."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def absolute_transformation_matrix(self) -> np.ndarray:
        return self._matrix


class _Raises:
    """Attribute-access stub whose every method call raises."""

    def __getattr__(self, name):
        def _fail(*args, **kwargs):
            raise RuntimeError("not available in this fake sim")

        return _fail


def _stage_template(asset_path: str):
    return types.SimpleNamespace(render_asset_fullpath=asset_path)


def _empty_manager():
    return types.SimpleNamespace(
        get_object_handles=lambda: [],
        get_object_by_handle=lambda h: None,
    )


def _rigid_manager(objs: dict):
    return types.SimpleNamespace(
        get_object_handles=lambda: list(objs),
        get_object_by_handle=lambda h: objs[h],
    )


def _ao_manager(aos: dict):
    return types.SimpleNamespace(
        get_object_handles=lambda: list(aos),
        get_object_by_handle=lambda h: aos[h],
    )


def _fake_sim(stage_asset: str, rom=None, aom=None):
    return types.SimpleNamespace(
        get_stage_initialization_template=lambda: _stage_template(stage_asset),
        get_rigid_object_manager=lambda: rom if rom is not None else _empty_manager(),
        get_articulated_object_manager=lambda: (
            aom if aom is not None else _empty_manager()
        ),
    )


def _make_ao(urdf_path: str, node: _MethodTransformNode):
    # semantic_id lives on creation_attributes, mirroring habitat's API
    # (managed articulated objects expose none directly).
    return types.SimpleNamespace(
        creation_attributes=types.SimpleNamespace(
            urdf_fullpath=urdf_path, semantic_id=7
        ),
        link_ids_to_object_ids={0: 42},
        motion_type=object(),  # not in _MT_MAP -> DYNAMIC
        object_id=40,
        awake=True,
        get_link_id_from_name=lambda name: {"door": 0}[name],
        get_link_scene_node=lambda link_id: node,
    )


class TestArticulatedLinkExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.floor = os.path.join(self._tmp.name, "floor.obj")
        _write_floor_obj(self.floor)
        mesh_path = os.path.join(self._tmp.name, "door.obj")
        _write_quad_obj(mesh_path)
        self.urdf_path = os.path.join(self._tmp.name, "cabinet.urdf")
        _write_urdf(self.urdf_path, "door.obj")

        self.world = np.eye(4, dtype=np.float32)
        self.world[:3, 3] = [1.0, 2.0, 3.0]
        self.node = _MethodTransformNode(self.world)
        self.sim = _fake_sim(
            self.floor,
            aom=_ao_manager({"cabinet_:0000": _make_ao(self.urdf_path, self.node)}),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_link_with_method_style_transform_is_extracted(self):
        model = extract_scene_model(self.sim, geometry="visual")

        # Instance 0 is the stage; instance 1 the articulated door link.
        self.assertEqual(model.num_instances, 2)
        np.testing.assert_array_equal(model.object_ids, [0, 42])
        np.testing.assert_array_equal(model.semantic_ids, [0, 7])
        self.assertEqual(model.objects[1].source, "articulated")
        np.testing.assert_allclose(model.transforms[1], self.world, atol=1e-6)

    def test_read_dynamic_transforms_sees_moved_link(self):
        model = extract_scene_model(self.sim, geometry="visual")

        moved = np.eye(4, dtype=np.float32)
        moved[:3, 3] = [4.0, 5.0, 6.0]
        self.node._matrix = moved

        changes = read_dynamic_transforms(self.sim, model, only_awake=False)
        self.assertEqual(set(changes), {1})
        np.testing.assert_allclose(changes[1], moved, atol=1e-6)


class TestExtractionFailLoud(unittest.TestCase):
    """Inputs that used to shrink the scene silently must now raise."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.floor = os.path.join(self._tmp.name, "floor.obj")
        _write_floor_obj(self.floor)

    def tearDown(self):
        self._tmp.cleanup()

    def test_stage_template_failure_propagates(self):
        # A failing stage read used to warn and drop the whole building shell.
        quad = os.path.join(self._tmp.name, "quad.obj")
        _write_quad_obj(quad)
        rigid = types.SimpleNamespace(
            creation_attributes=types.SimpleNamespace(render_asset_fullpath=quad),
            scale=np.ones(3),
            transformation=np.eye(4, dtype=np.float32),
            object_id=10,
            semantic_id=3,
            motion_type=object(),
        )
        sim = types.SimpleNamespace(
            get_stage_initialization_template=_Raises().get_stage_initialization_template,
            get_rigid_object_manager=lambda: _rigid_manager({"a": rigid}),
            get_articulated_object_manager=lambda: _empty_manager(),
        )
        with self.assertRaises(RuntimeError):
            extract_scene_model(sim, geometry="visual")

    def test_unloadable_rigid_asset_raises(self):
        # A rigid object whose mesh cannot be loaded used to vanish silently.
        rigid = types.SimpleNamespace(
            creation_attributes=types.SimpleNamespace(
                render_asset_fullpath=os.path.join(self._tmp.name, "no_such.glb")
            ),
            scale=np.ones(3),
            transformation=np.eye(4, dtype=np.float32),
            object_id=10,
            semantic_id=3,
            motion_type=object(),
        )
        sim = _fake_sim(self.floor, rom=_rigid_manager({"a": rigid}))
        with self.assertRaisesRegex(RuntimeError, "no_such.glb"):
            extract_scene_model(sim, geometry="visual")

    def test_missing_articulated_urdf_raises(self):
        node = _MethodTransformNode(np.eye(4, dtype=np.float32))
        ao = _make_ao(os.path.join(self._tmp.name, "gone.urdf"), node)
        sim = _fake_sim(self.floor, aom=_ao_manager({"cab": ao}))
        with self.assertRaises(RuntimeError):
            extract_scene_model(sim, geometry="visual")

    def test_articulated_urdf_without_mesh_visuals_raises(self):
        # habitat loaded this AO from its URDF, so extracting zero mesh
        # visuals from the same file means the AO would silently disappear.
        urdf = os.path.join(self._tmp.name, "empty.urdf")
        with open(urdf, "w") as f:
            f.write('<robot name="empty"><link name="door"/></robot>')
        node = _MethodTransformNode(np.eye(4, dtype=np.float32))
        ao = _make_ao(urdf, node)
        sim = _fake_sim(self.floor, aom=_ao_manager({"cab": ao}))
        with self.assertRaises(RuntimeError):
            extract_scene_model(sim, geometry="visual")


if __name__ == "__main__":
    unittest.main()
