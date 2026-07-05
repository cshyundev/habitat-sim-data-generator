"""
Imports the sensor classes that ship with this package so their
`@register_sensor` decorators run and populate the registry.

A third-party sensor plugin doesn't need this module -- it only needs to
decorate its own BaseSensor subclass with `@register_sensor("its_type")` and
be imported before a SensorSuite is constructed. This module just guarantees
the built-in types (lidar3d, laser2d, camera, imu) are always available.
"""
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D
from src.sensors.laser2d.ideal_laser import IdealLaser2D
from src.sensors.camera.camera import CameraSensor
from src.sensors.imu.ideal_imu import IdealIMU

__all__ = ["IdealLiDAR3D", "IdealLaser2D", "CameraSensor", "IdealIMU"]
