from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from src.planners.global_planning.base import BaseGlobalPlanner
from src.planners.global_planning.params import ZigzagCoverageParams
from src.planners.global_planning.zigzag_coverage import ZigzagCoveragePlanner
from src.planners.local_planning.base import BaseLocalPlanner
from src.planners.local_planning.differential_drive import DifferentialDriveLocalPlanner
from src.planners.local_planning.params import DifferentialDriveParams
from src.robot_config import ConfigError

if TYPE_CHECKING:
    from src.runtime_config import PlannerConfig


# A builder takes the planner's *typed params* (parsed once at the boundary) and
# returns a planner. A params-parser turns the raw config dict into those typed
# params -- the only thing that touches the dict, and only during boundary parse.
GlobalPlannerBuilder = Callable[[Any], BaseGlobalPlanner]
LocalPlannerBuilder = Callable[[Any], BaseLocalPlanner]
ParamsParser = Callable[[dict], Any]


@dataclass(frozen=True)
class _PlannerEntry:
    builder: Callable[[Any], Any]
    params_parser: ParamsParser


_GLOBAL_PLANNERS: Dict[str, _PlannerEntry] = {}
_LOCAL_PLANNERS: Dict[str, _PlannerEntry] = {}


def _identity_params(config: dict) -> dict:
    return config


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


def register_global_planner(
    type_name: str,
    builder: GlobalPlannerBuilder,
    params_parser: ParamsParser = _identity_params,
) -> None:
    key = str(type_name).lower()
    existing = _GLOBAL_PLANNERS.get(key)
    if existing is not None and existing.builder is not builder:
        raise ValueError(f"Global planner type '{key}' is already registered.")
    _GLOBAL_PLANNERS[key] = _PlannerEntry(builder, params_parser)


def register_local_planner(
    type_name: str,
    builder: LocalPlannerBuilder,
    params_parser: ParamsParser = _identity_params,
) -> None:
    key = str(type_name).lower()
    existing = _LOCAL_PLANNERS.get(key)
    if existing is not None and existing.builder is not builder:
        raise ValueError(f"Local planner type '{key}' is already registered.")
    _LOCAL_PLANNERS[key] = _PlannerEntry(builder, params_parser)


def available_global_planners() -> tuple[str, ...]:
    return tuple(sorted(_GLOBAL_PLANNERS))


def available_local_planners() -> tuple[str, ...]:
    return tuple(sorted(_LOCAL_PLANNERS))


def _require_global(key: str) -> _PlannerEntry:
    entry = _GLOBAL_PLANNERS.get(key)
    if entry is None:
        raise ConfigError(
            f"Unknown global planner type '{key}'. "
            f"Available: {list(available_global_planners())}"
        )
    return entry


def _require_local(key: str) -> _PlannerEntry:
    entry = _LOCAL_PLANNERS.get(key)
    if entry is None:
        raise ConfigError(
            f"Unknown local planner type '{key}'. "
            f"Available: {list(available_local_planners())}"
        )
    return entry


def parse_global_params(config: dict) -> Any:
    """Boundary parse: raw dict -> the chosen global planner's typed params."""
    return _require_global(global_planner_type(config)).params_parser(config)


def parse_local_params(config: dict) -> Any:
    """Boundary parse: raw dict -> the chosen local planner's typed params."""
    return _require_local(local_planner_type(config)).params_parser(config)


def create_global_planner(config: dict) -> BaseGlobalPlanner:
    """Convenience boundary factory: parse the dict and build in one step."""
    entry = _require_global(global_planner_type(config))
    return entry.builder(entry.params_parser(config))


def create_local_planner(config: dict) -> BaseLocalPlanner:
    """Convenience boundary factory: parse the dict and build in one step."""
    entry = _require_local(local_planner_type(config))
    return entry.builder(entry.params_parser(config))


def build_planners(
    planner_config: "PlannerConfig",
) -> Tuple[BaseGlobalPlanner, BaseLocalPlanner]:
    """Build both planners from an already-parsed :class:`PlannerConfig` slice --
    the downstream path (no raw dict, no re-parse)."""
    global_planner = _require_global(planner_config.global_type).builder(
        planner_config.global_params
    )
    local_planner = _require_local(planner_config.local_type).builder(
        planner_config.local_params
    )
    return global_planner, local_planner


def _build_zigzag(params: ZigzagCoverageParams) -> BaseGlobalPlanner:
    return ZigzagCoveragePlanner(params)


def _build_differential_drive(params: DifferentialDriveParams) -> BaseLocalPlanner:
    return DifferentialDriveLocalPlanner(params)


register_global_planner("zigzag", _build_zigzag, ZigzagCoverageParams.from_config)
register_local_planner(
    "differential_drive", _build_differential_drive, DifferentialDriveParams.from_config
)
