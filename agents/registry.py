from typing import Dict, Type
from agents.base_agent import BaseRLAgent


# registry maps algorithm name -> agent class
# populated as agents are implemented in Phase 4
_REGISTRY: Dict[str, Type[BaseRLAgent]] = {}


def register(name: str):
    """Decorator to register an agent class by algorithm name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_agent_class(name: str) -> Type[BaseRLAgent]:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown agent algorithm: '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_agents() -> list:
    return list(_REGISTRY.keys())