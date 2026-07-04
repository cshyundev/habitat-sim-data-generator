"""ROS 2 message definitions for every schema this project writes to MCAP.

Each entry is real ROS 2 (Humble) message text in the concatenated form
rosbag2 uses: the top-level type's fields, followed by ``===`` separators and
``MSG: pkg/Type`` blocks for every nested type it references (nested field
types use the short ``pkg/Type`` form, not ``pkg/msg/Type`` -- that's what the
vendored rosidl_adapter parser in ``mcap_ros2`` expects). ``builtin_interfaces/
Time`` and ``Duration`` are predefined by the library and intentionally not
redeclared here.

``McapExporter`` (see ``src/utils/export.py``) looks up this dict by schema
name and hands the text to ``mcap_ros2.writer.Writer.register_msgdef``, which
writes it as the channel's Schema record -- making the MCAP self-describing
and decodable by any ROS 2 / Foxglove / rosbag2 tool, not just this repo's own
reader.

The two ``habitat_msgs/msg/Detection*Array`` schemas are NOT real ROS 2
messages -- there is no standard message for instance-id + class-name boxes.
They are named under a project-owned package instead of borrowing
``vision_msgs`` names that don't match this payload.
"""

_HEADER = """\
builtin_interfaces/Time stamp
string frame_id
"""

_POINT = """\
float64 x
float64 y
float64 z
"""

_QUATERNION = """\
float64 x
float64 y
float64 z
float64 w
"""

_VECTOR3 = """\
float64 x
float64 y
float64 z
"""

_POSE = """\
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
"""

_COLOR_RGBA = """\
float32 r
float32 g
float32 b
float32 a
"""

_SEP = "=" * 80


def _concat(top_fields: str, *sub_msgdefs: "tuple[str, str]") -> str:
    """Joins a top-level field block with ``(name, fields)`` nested msgdefs."""
    parts = [top_fields.rstrip() + "\n"]
    for name, fields in sub_msgdefs:
        parts.append(_SEP)
        parts.append(f"MSG: {name}")
        parts.append(fields.rstrip())
    return "\n".join(parts) + "\n"


_GEOMETRY_COMMON = [
    ("geometry_msgs/Point", _POINT),
    ("geometry_msgs/Quaternion", _QUATERNION),
    ("geometry_msgs/Vector3", _VECTOR3),
    ("geometry_msgs/Pose", _POSE),
]

MSGDEFS = {
    "geometry_msgs/msg/PoseStamped": _concat(
        """\
std_msgs/Header header
geometry_msgs/Pose pose
""",
        ("std_msgs/Header", _HEADER),
        *_GEOMETRY_COMMON,
    ),
    "sensor_msgs/msg/PointCloud2": _concat(
        """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
""",
        ("std_msgs/Header", _HEADER),
        (
            "sensor_msgs/PointField",
            """\
uint8 INT8 = 1
uint8 UINT8 = 2
uint8 INT16 = 3
uint8 UINT16 = 4
uint8 INT32 = 5
uint8 UINT32 = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8
string name
uint32 offset
uint8 datatype
uint32 count
""",
        ),
    ),
    "sensor_msgs/msg/LaserScan": _concat(
        """\
std_msgs/Header header
float32 angle_min
float32 angle_max
float32 angle_increment
float32 time_increment
float32 scan_time
float32 range_min
float32 range_max
float32[] ranges
float32[] intensities
""",
        ("std_msgs/Header", _HEADER),
    ),
    "sensor_msgs/msg/Image": _concat(
        """\
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
""",
        ("std_msgs/Header", _HEADER),
    ),
    "sensor_msgs/msg/Imu": _concat(
        """\
std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
""",
        ("std_msgs/Header", _HEADER),
        ("geometry_msgs/Quaternion", _QUATERNION),
        ("geometry_msgs/Vector3", _VECTOR3),
    ),
    "nav_msgs/msg/OccupancyGrid": _concat(
        """\
std_msgs/Header header
nav_msgs/MapMetaData info
int8[] data
""",
        ("std_msgs/Header", _HEADER),
        (
            "nav_msgs/MapMetaData",
            """\
builtin_interfaces/Time map_load_time
float32 resolution
uint32 width
uint32 height
geometry_msgs/Pose origin
""",
        ),
        *_GEOMETRY_COMMON,
    ),
    "visualization_msgs/msg/MarkerArray": _concat(
        """\
visualization_msgs/Marker[] markers
""",
        (
            "visualization_msgs/Marker",
            """\
uint8 ARROW=0
uint8 CUBE=1
uint8 SPHERE=2
uint8 CYLINDER=3
uint8 LINE_STRIP=4
uint8 LINE_LIST=5
uint8 CUBE_LIST=6
uint8 SPHERE_LIST=7
uint8 POINTS=8
uint8 TEXT_VIEW_FACING=9
uint8 MESH_RESOURCE=10
uint8 TRIANGLE_LIST=11
uint8 ADD=0
uint8 MODIFY=1
uint8 DELETE=2
uint8 DELETEALL=3
std_msgs/Header header
string ns
int32 id
int32 type
int32 action
geometry_msgs/Pose pose
geometry_msgs/Vector3 scale
std_msgs/ColorRGBA color
builtin_interfaces/Duration lifetime
bool frame_locked
geometry_msgs/Point[] points
std_msgs/ColorRGBA[] colors
string text
string mesh_resource
bool mesh_use_embedded_materials
""",
        ),
        ("std_msgs/Header", _HEADER),
        ("std_msgs/ColorRGBA", _COLOR_RGBA),
        *_GEOMETRY_COMMON,
    ),
    "tf2_msgs/msg/TFMessage": _concat(
        """\
geometry_msgs/TransformStamped[] transforms
""",
        (
            "geometry_msgs/TransformStamped",
            """\
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
""",
        ),
        (
            "geometry_msgs/Transform",
            """\
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
""",
        ),
        ("std_msgs/Header", _HEADER),
        ("geometry_msgs/Vector3", _VECTOR3),
        ("geometry_msgs/Quaternion", _QUATERNION),
    ),
    # Project-owned schemas (not real ROS 2 messages -- see module docstring).
    "habitat_msgs/msg/Detection2DArray": _concat(
        """\
std_msgs/Header header
habitat_msgs/Detection2D[] detections
""",
        ("std_msgs/Header", _HEADER),
        (
            "habitat_msgs/Detection2D",
            """\
int32 instance_id
int32 class_id
string class_name
int32[4] xyxy
""",
        ),
    ),
    "habitat_msgs/msg/Detection3DArray": _concat(
        """\
std_msgs/Header header
habitat_msgs/Detection3D[] detections
""",
        ("std_msgs/Header", _HEADER),
        (
            "habitat_msgs/Detection3D",
            """\
int32 instance_id
int32 class_id
string class_name
geometry_msgs/Point center
geometry_msgs/Vector3 half_extents
geometry_msgs/Quaternion orientation
string frame
""",
        ),
        ("geometry_msgs/Point", _POINT),
        ("geometry_msgs/Vector3", _VECTOR3),
        ("geometry_msgs/Quaternion", _QUATERNION),
    ),
}
