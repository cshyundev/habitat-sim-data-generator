from __future__ import annotations

from typing import Callable, Dict

from src.planners.global_planning.base import BaseGlobalPlanner
from src.planners.global_planning.params import ZigzagCoverageParams
from src.planners.global_planning.zigzag_coverage import ZigzagCoveragePlanner
from src.planners.local_planning.base import BaseLocalPlanner
from src.planners.local_planning.differential_drive import DifferentialDriveLocalPlanner
from src.planners.local_planning.params import DifferentialDriveParams
from src.robot_config import ConfigError


GlobalPlannerBuilder = Callable[[dict], BaseGlobalPlanner]
LocalPlannerBuilder = Callable[[dict], BaseLocalPlanner]

_GLOBAL_PLANNERS: Dict[str, GlobalPlannerBuilder] = {}
_LOCAL_PLANNERS: Dict[str, LocalPlannerBuilder] = {}


def _planner_section(config: dict, key: str, legacy_key: str | None = None) -> dict:
    planner = config.get("planner", {}) or {}
    if isinstance(planner, dict) and key in planner:
        section = planner.get(key) or {}
    elif legacy_key is not None:
        section = config.get(legacy_key, {}) or {}
    else:
        section = planner
    if not isinstance(section, dict):
        raise ConfigError(f"planner.{key}: must be a mapping.")
    return section


def global_planner_type(config: dict) -> str:
    section = _planner_section(config, "global")
    return str(section.get("type", "zigzag")).lower()


def local_planner_type(config: dict) -> str:
    planner = config.get("planner", {}) or {}
    if isinstance(planner, dict) and "local" in planner:
        section = planner.get("local") or {}
    else:
        section = config.get("local_planner", {}) or {}
    if not isinstance(section, dict):
        raise ConfigError("planner.local: must be a mapping.")
    return str(section.get("type", "differential_drive")).lower()


def register_global_planner(type_name: str, builder: GlobalPlannerBuilder) -> None:
    key = str(type_name).lower()
    existing = _GLOBAL_PLANNERS.get(key)
    if existing is not None and existing is not builder:
        raise ValueError(f"Global planner type '{key}' is already registered.")
    _GLOBAL_PLANNERS[key] = builder


def register_local_planner(type_name: str, builder: LocalPlannerBuilder) -> None:
    key = str(type_name).lower()
    existing = _LOCAL_PLANNERS.get(key)
    if existing is not None and existing is not builder:
        raise ValueError(f"Local planner type '{key}' is already registered.")
    _LOCAL_PLANNERS[key] = builder


def available_global_planners() -> tuple[str, ...]:
    return tuple(sorted(_GLOBAL_PLANNERS))


def available_local_planners() -> tuple[str, ...]:
    return tuple(sorted(_LOCAL_PLANNERS))


def create_global_planner(config: dict) -> BaseGlobalPlanner:
    key = global_planner_type(config)
    builder = _GLOBAL_PLANNERS.get(key)
    if builder is None:
        raise ConfigError(
            f"Unknown global planner type '{key}'. "
            f"Available: {list(available_global_planners())}"
        )
    return builder(config)


def create_local_planner(config: dict) -> BaseLocalPlanner:
    key = local_planner_type(config)
    builder = _LOCAL_PLANNERS.get(key)
    if builder is None:
        raise ConfigError(
            f"Unknown local planner type '{key}'. "
            f"Available: {list(available_local_planners())}"
        )
    return builder(config)


def _build_zigzag(config: dict) -> BaseGlobalPlanner:
    return ZigzagCoveragePlanner(ZigzagCoverageParams.from_config(config))


def _build_differential_drive(config: dict) -> BaseLocalPlanner:
    return DifferentialDriveLocalPlanner(DifferentialDriveParams.from_config(config))


register_global_planner("zigzag", _build_zigzag)
register_local_planner("differential_drive", _build_differential_drive)
