from typing import Callable, Dict, List, Type

from src.sensors.base_sensor import BaseSensor

_REGISTRY: Dict[str, Type[BaseSensor]] = {}


def register_sensor(type_name: str) -> Callable[[Type[BaseSensor]], Type[BaseSensor]]:
    """
    Class decorator that registers a BaseSensor subclass under a config
    "type" string, so SensorSuite can build it by lookup instead of an
    if/elif chain. Adding a new sensor type (in this package or a plugin
    module) only requires this decorator -- no other file needs to change.
    """
    def _decorate(cls: Type[BaseSensor]) -> Type[BaseSensor]:
        existing = _REGISTRY.get(type_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Sensor type '{type_name}' is already registered to "
                f"{existing.__name__}; cannot also register {cls.__name__}."
            )
        _REGISTRY[type_name] = cls
        return cls
    return _decorate


def get_sensor_class(type_name: str) -> Type[BaseSensor]:
    """
    Looks up the BaseSensor subclass registered for `type_name`.

    Raises:
        KeyError: if no sensor class is registered under that name.
    """
    try:
        return _REGISTRY[type_name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"Unsupported sensor type '{type_name}'. Available: {available}."
        ) from None


def registered_sensor_types() -> List[str]:
    """Returns the currently registered config "type" strings, sorted."""
    return sorted(_REGISTRY)
