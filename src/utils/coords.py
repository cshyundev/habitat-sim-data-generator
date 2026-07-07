import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
# pyrefly: ignore [missing-import]
from src.datatypes.pose import Pose3D
from src.datatypes.bbox import OBB3D
from src.utils.geometry import matrix_to_quaternion, quaternion_to_matrix, rpy_to_matrix

logger = logging.getLogger(__name__)

# Habitat -> ROS basis change (X_ros=-Z_hab, Y_ros=-X_hab, Z_ros=Y_hab). Applied
# on the LEFT to a box rotation so its axes (and directional half-extents) map
# correctly; conjugation (as used for robot/sensor Pose3D orientation, whose
# own local frame also follows the Habitat forward/up/right convention) would
# mis-order an OBB's half-extents since a box's local axes are arbitrary
# object geometry, not a Habitat-convention body frame.
_R_HAB_TO_ROS = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def habitat_to_ros_obb(obb: OBB3D) -> OBB3D:
    """Convert a world-frame OBB3D from Habitat coordinates to ROS coordinates.

    Args:
        obb: Habitat-frame oriented bounding box.

    Returns:
        ROS-frame oriented bounding box with ``frame="map"``.
    """
    R_ros = _R_HAB_TO_ROS @ quaternion_to_matrix(obb.quat_xyzw)
    center_ros = _R_HAB_TO_ROS @ np.asarray(obb.center, dtype=np.float64)
    quat_ros = matrix_to_quaternion(R_ros)
    return OBB3D(
        instance_id=obb.instance_id,
        class_id=obb.class_id,
        class_name=obb.class_name,
        center=center_ros.astype(np.float32),
        half_extents=np.asarray(obb.half_extents, dtype=np.float32),
        quat_xyzw=quat_ros.astype(np.float32),
        frame="map",
    )


def habitat_to_ros_pose(pose: Pose3D) -> Pose3D:
    """Convert a Habitat-frame pose to ROS coordinates.

    Args:
        pose: Pose in Habitat coordinates.

    Returns:
        Equivalent pose in ROS coordinates.
    """
    return Pose3D(
        position=habitat_to_ros_position(pose.position),
        orientation=habitat_to_ros_quaternion(pose.orientation)
    )

def habitat_to_ros_position(p: np.ndarray) -> np.ndarray:
    """Convert a position from Habitat coordinates to ROS coordinates.

    Args:
        p: Habitat position ``[x, y, z]``.

    Returns:
        ROS position ``[x, y, z]``.
    """
    return np.array([-p[2], -p[0], p[1]], dtype=p.dtype)

def habitat_to_ros_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion from Habitat coordinates to ROS coordinates.

    Args:
        q: Quaternion in ``[x, y, z, w]`` order.

    Returns:
        ROS-frame quaternion in ``[x, y, z, w]`` order.
    """
    return np.array([-q[2], -q[0], q[1], q[3]], dtype=q.dtype)

def ros_to_habitat_position(p: np.ndarray) -> np.ndarray:
    """Convert a ROS/URDF position to Habitat coordinates.

    Args:
        p: ROS position ``[x, y, z]``.

    Returns:
        Habitat position ``[x, y, z]``.
    """
    return np.array([-p[1], p[2], -p[0]], dtype=p.dtype)

def ros_to_habitat_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a ROS/URDF quaternion to Habitat coordinates.

    Args:
        q: Quaternion in ``[x, y, z, w]`` order.

    Returns:
        Habitat-frame quaternion in ``[x, y, z, w]`` order.
    """
    return np.array([-q[1], q[2], -q[0], q[3]], dtype=q.dtype)

def habitat_to_ros_pointcloud(pc: np.ndarray) -> np.ndarray:
    """Convert an ``N x 3`` point cloud from Habitat to ROS coordinates.

    Args:
        pc: Point cloud in Habitat coordinates.

    Returns:
        Point cloud in ROS coordinates.
    """
    if pc.shape[0] == 0:
        return pc
    
    ros_pc = np.empty_like(pc)
    ros_pc[:, 0] = -pc[:, 2] # X_ros = -Z_hab
    ros_pc[:, 1] = -pc[:, 0] # Y_ros = -X_hab
    ros_pc[:, 2] = pc[:, 1]  # Z_ros = Y_hab
    return ros_pc


def convert_occupancy_grid_to_ros(occ_grid) -> Tuple[Pose3D, np.ndarray]:
    """Convert an occupancy grid to ROS map-server conventions.

    Args:
        occ_grid: Simulator/planner occupancy grid.

    Returns:
        Pair of ROS-frame origin pose and ROS occupancy data.
    """
    from src.datatypes.map import GRID_2D_FREE, GRID_2D_OCCUPIED
    
    ros_map_data = np.full(occ_grid.data.shape, -1, dtype=np.int8)
    ros_map_data[occ_grid.data == GRID_2D_FREE] = 0
    ros_map_data[occ_grid.data == GRID_2D_OCCUPIED] = 100
    
    # Flip vertically to convert from image-space (top-left row 0) to ROS-space (bottom-left row 0)
    ros_map_data_flipped = np.flipud(ros_map_data)
    
    # Transform origin pose to ROS standard coordinates
    pos_ros = habitat_to_ros_position(occ_grid.origin.position)
    q_ros = habitat_to_ros_quaternion(occ_grid.origin.orientation)
    origin_pose_ros = Pose3D(pos_ros, q_ros)
    
    return origin_pose_ros, ros_map_data_flipped


def parse_urdf_visuals(urdf_path: str) -> Dict[str, List[Dict[str, object]]]:
    """Parse mesh visual entries from a URDF.

    Args:
        urdf_path: Absolute or relative URDF path.

    Returns:
        Mapping from link name to visual mesh descriptors.
    """
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except (OSError, ET.ParseError) as e:
        logger.warning("Failed to parse URDF xml at %s: %s", urdf_path, e)
        return {}
    
    link_visuals: Dict[str, List[Dict[str, object]]] = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        visuals: List[Dict[str, object]] = []
        for visual in link.findall('visual'):
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]
            origin = visual.find('origin')
            if origin is not None:
                xyz_str = origin.get('xyz')
                if xyz_str:
                    origin_xyz = [float(x) for x in xyz_str.split()]
                rpy_str = origin.get('rpy')
                if rpy_str:
                    origin_rpy = [float(x) for x in rpy_str.split()]
            
            geom = visual.find('geometry')
            if geom is not None:
                mesh = geom.find('mesh')
                if mesh is not None:
                    filename = mesh.get('filename')
                    scale_str = mesh.get('scale')
                    scale = [1.0, 1.0, 1.0]
                    if scale_str:
                        scale_str = scale_str.replace(',', ' ')
                        scale = [float(x) for x in scale_str.split()]
                        if len(scale) != 3:
                            scale = [1.0, 1.0, 1.0]
                    visuals.append({
                        'filename': filename,
                        'scale': scale,
                        'origin_xyz': origin_xyz,
                        'origin_rpy': origin_rpy
                    })
        link_visuals[link_name] = visuals
    return link_visuals


def resolve_urdf_path(urdf_rel: str, scene_dataset: str) -> Optional[str]:
    """Resolve a URDF path against known ReplicaCAD dataset locations.

    Args:
        urdf_rel: URDF path from habitat metadata.
        scene_dataset: Scene dataset config path.

    Returns:
        Existing absolute path, or ``None`` when not found.
    """
    import os
    real_dataset_dir = os.path.dirname(os.path.realpath(scene_dataset))
    path_attempts = [
        os.path.abspath(urdf_rel),
        os.path.abspath(os.path.join("habitat-sim/data/replica_cad", urdf_rel)),
        os.path.abspath(os.path.join("habitat-sim/data/versioned_data/replica_cad_dataset", urdf_rel)),
    ]
    if urdf_rel.startswith("habitat-sim/data/replica_cad/"):
        stripped = urdf_rel[len("habitat-sim/data/replica_cad/"):]
        path_attempts.append(os.path.abspath(os.path.join("habitat-sim/data/versioned_data/replica_cad_dataset", stripped)))
        path_attempts.append(os.path.abspath(os.path.join("habitat-sim/data/replica_cad", stripped)))
        
    for p in path_attempts:
        if os.path.exists(p):
            return p
    return None


def extract_visual_map_as_markers(
    sim,
    scene_dataset_config_file: str,
) -> List[Dict[str, object]]:
    """Extract visual scene geometry as marker dictionaries.

    Args:
        sim: Running habitat simulator.
        scene_dataset_config_file: Validated scene-dataset config path used to
            resolve mesh asset paths.

    Returns:
        Marker dictionaries compatible with the existing MCAP/export helpers.
    """
    import os
    import trimesh

    scene_dataset = scene_dataset_config_file
    active_stage_attr = sim.get_stage_initialization_template()
    
    def resolve_mesh_path(handle: str) -> str:
        """Resolve a mesh asset handle against the dataset directory."""
        real_dataset_dir = os.path.dirname(os.path.realpath(scene_dataset))
        clean_handle = handle
        if clean_handle.startswith("../../"):
            clean_handle = clean_handle[6:]
        return os.path.abspath(os.path.join(real_dataset_dir, clean_handle))
        
    stage_path = resolve_mesh_path(active_stage_attr.render_asset_handle)
    markers_list: List[Dict[str, object]] = []
    marker_id_counter = 0
    
    # 1. Load stage mesh
    if os.path.exists(stage_path):
        logger.info("3D map: loading stage mesh %s", os.path.basename(stage_path))
        try:
            m_stage = trimesh.load(stage_path)
            if isinstance(m_stage, trimesh.Scene):
                m_stage = m_stage.to_geometry()
            m_stage.visual = m_stage.visual.to_color()
            
            stage_verts_ros = habitat_to_ros_pointcloud(np.array(m_stage.vertices, dtype=np.float64))
            stage_faces = np.array(m_stage.faces, dtype=np.int32)
            # Flip winding order due to reflection coordinate basis transform
            stage_faces = stage_faces[:, [0, 2, 1]]
            
            markers_list.append({
                'ns': 'stage',
                'id': marker_id_counter,
                'type': 11, # TRIANGLE_LIST
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'scale': np.array([1.0, 1.0, 1.0]),
                'vertices': stage_verts_ros,
                'indices': stage_faces.tolist(),
                'vertex_colors': m_stage.visual.vertex_colors,
                'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0
            })
            marker_id_counter += 1
        except Exception as e:
            logger.warning("Failed to load stage mesh: %s", e)
            
    # 2. Load rigid objects meshes
    rigid_obj_mgr = sim.get_rigid_object_manager()
    logger.info("3D map: loading rigid object meshes (%d total)", len(rigid_obj_mgr.get_object_handles()))
    for handle in rigid_obj_mgr.get_object_handles():
        obj = rigid_obj_mgr.get_object_by_handle(handle)
        template = obj.creation_attributes
        obj_path = resolve_mesh_path(template.render_asset_handle)
        if os.path.exists(obj_path):
            try:
                m_obj = trimesh.load(obj_path)
                if isinstance(m_obj, trimesh.Scene):
                    m_obj = m_obj.to_geometry()
                m_obj.visual = m_obj.visual.to_color()
                
                obj_verts_ros = habitat_to_ros_pointcloud(np.array(m_obj.vertices, dtype=np.float64))
                obj_faces = np.array(m_obj.faces, dtype=np.int32)
                # Flip winding order due to reflection coordinate basis transform
                obj_faces = obj_faces[:, [0, 2, 1]]
                
                # Convert object pose to ROS coordinate frame
                pos_ros = habitat_to_ros_position(np.array(obj.translation))
                q_imag = obj.rotation.vector
                q_real = obj.rotation.scalar
                q_ros = habitat_to_ros_quaternion(np.array([q_imag[0], q_imag[1], q_imag[2], q_real]))
                
                markers_list.append({
                    'ns': 'object',
                    'id': marker_id_counter,
                    'type': 11, # TRIANGLE_LIST
                    'position': pos_ros,
                    'orientation': q_ros,
                    'scale': np.array([1.0, 1.0, 1.0]),
                    'vertices': obj_verts_ros,
                    'indices': obj_faces.tolist(),
                    'vertex_colors': m_obj.visual.vertex_colors,
                    'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0
                })
                marker_id_counter += 1
            except Exception as e:
                logger.debug("Skipping rigid object mesh %s: %s", handle, e)

    # 3. Load articulated objects meshes
    ao_mgr = sim.get_articulated_object_manager()
    logger.info("3D map: loading articulated object meshes (%d total)", len(ao_mgr.get_object_handles()))
    for handle in ao_mgr.get_object_handles():
        ao = ao_mgr.get_object_by_handle(handle)
        urdf_rel = ao.creation_attributes.urdf_fullpath
        urdf_abs = resolve_urdf_path(urdf_rel, scene_dataset)
        if urdf_abs and os.path.exists(urdf_abs):
            try:
                visuals = parse_urdf_visuals(urdf_abs)
                
                # Build link name to ID mapping
                link_name_to_id = {}
                for link_id in ao.get_link_ids():
                    link_name_to_id[ao.get_link_name(link_id)] = link_id
                
                for lname, vis_list in visuals.items():
                    # Find scene node
                    if lname in link_name_to_id:
                        node = ao.get_link_scene_node(link_name_to_id[lname])
                    else:
                        node = ao.root_scene_node
                    
                    # Absolute transformation of this link in habitat coordinates
                    T_link_world = np.array(node.absolute_transformation())
                    
                    for vis in vis_list:
                        mesh_dir = os.path.dirname(urdf_abs)
                        mesh_abs = os.path.abspath(os.path.join(mesh_dir, vis['filename']))
                        if os.path.exists(mesh_abs):
                            try:
                                m_link = trimesh.load(mesh_abs)
                                if isinstance(m_link, trimesh.Scene):
                                    m_link = m_link.to_geometry()
                                m_link.visual = m_link.visual.to_color()
                                
                                # Scale vertices
                                scale = np.array(vis['scale'])
                                m_link.vertices = m_link.vertices * scale
                                
                                # Visual origin transform
                                T_vis = np.identity(4)
                                T_vis[:3, :3] = rpy_to_matrix(vis['origin_rpy'])
                                T_vis[:3, 3] = vis['origin_xyz']
                                m_link.apply_transform(T_vis)
                                
                                # Link world transform
                                m_link.apply_transform(T_link_world)
                                
                                # Convert to ROS coordinate frame
                                verts_ros = habitat_to_ros_pointcloud(np.array(m_link.vertices, dtype=np.float64))
                                faces_ros = np.array(m_link.faces, dtype=np.int32)[:, [0, 2, 1]]
                                
                                markers_list.append({
                                    'ns': f'articulated_{handle.split(":")[0]}',
                                    'id': marker_id_counter,
                                    'type': 11, # TRIANGLE_LIST
                                    'position': np.array([0.0, 0.0, 0.0]),
                                    'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                                    'scale': np.array([1.0, 1.0, 1.0]),
                                    'vertices': verts_ros,
                                    'indices': faces_ros.tolist(),
                                    'vertex_colors': m_link.visual.vertex_colors,
                                    'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0
                                })
                                marker_id_counter += 1
                            except Exception as e:
                                logger.warning("Failed to process visual mesh %s: %s", vis["filename"], e)
            except Exception as e:
                logger.warning("Failed to process articulated object %s: %s", handle, e)
                
    return markers_list
