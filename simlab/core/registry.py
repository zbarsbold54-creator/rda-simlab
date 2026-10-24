"""SimLab registry module - @register(\"drive\",\"name\") + get(\"drive\",\"name\")."""

from __future__ import annotations

import builtins
import importlib
import pkgutil
from typing import Any


class Registry:
    """Central registry for managing plugins and registered types.

    Supports decorator-based registration and lookup by type and name.
    Used for registering drives, policies, worlds, entities, sensors, and metrics.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registrations: dict[str, dict[str, type[Any]]] = {}

    def register(self, type_name: str, name: str) -> callable:
        """Decorator to register a class with a type name and identifier.

        Args:
            type_name: Category of the type (e.g., "drive", "policy", "world").
            name: Identifier name for the registration.

        Returns:
            Decorator function that registers the decorated class.
        """

        def decorator(cls: type[Any]) -> type[Any]:
            if type_name not in self._registrations:
                self._registrations[type_name] = {}

            self._registrations[type_name][name] = cls
            return cls

        return decorator

    def get(self, type_name: str, name: str) -> type[Any] | None:
        """Get a registered class by type name and identifier.

        Args:
            type_name: Category of the type (e.g., "drive", "policy", "world").
            name: Identifier name of the registration.

        Returns:
            Registered class or None if not found.
        """
        if type_name in self._registrations and name in self._registrations[type_name]:
            return self._registrations[type_name][name]
        return None

    def list(self, type_name: str) -> builtins.list[str]:
        """List all registered names for a given type.

        Args:
            type_name: Category of the type (e.g., "drive", "policy", "world").

        Returns:
            List of registered names.
        """
        if type_name in self._registrations:
            return list(self._registrations[type_name].keys())
        return []

    def all(self) -> dict[str, dict[str, type[Any]]]:
        """Get all registrations.

        Returns:
            Dictionary mapping type names to name->class mappings.
        """
        return self._registrations.copy()

    def exists(self, type_name: str, name: str) -> bool:
        """Check if a registration exists.

        Args:
            type_name: Category of the type.
            name: Identifier name.

        Returns:
            True if the registration exists, False otherwise.
        """
        return (
            type_name in self._registrations and name in self._registrations[type_name]
        )


# Global registry instance
_global_registry: Registry | None = None


def get_global_registry() -> Registry:
    """Get the global registry instance.

    Returns:
        The global Registry instance, creating it if necessary.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = Registry()
    return _global_registry


def register(type_name: str, name: str) -> callable:
    """Convenience function to register a class using the global registry.

    Args:
        type_name: Category of the type (e.g., "drive", "policy", "world").
        name: Identifier name for the registration.

    Returns:
        Decorator function that registers the decorated class.
    """
    return get_global_registry().register(type_name, name)


def get(type_name: str, name: str) -> type[Any] | None:
    """Convenience function to get a class from the global registry.

    Args:
        type_name: Category of the type (e.g., "drive", "policy", "world").
        name: Identifier name of the registration.

    Returns:
        Registered class or None if not found.
    """
    return get_global_registry().get(type_name, name)


def list_(type_name: str) -> list[str]:
    """Convenience function to list registrations from the global registry.

    Args:
        type_name: Category of the type (e.g., "drive", "policy", "world").

    Returns:
        List of registered names.
    """
    return get_global_registry().list(type_name)


def all_() -> dict[str, dict[str, type[Any]]]:
    """Convenience function to get all registrations from the global registry.

    Returns:
        Dictionary mapping type names to name->class mappings.
    """
    return get_global_registry().all()


def load_plugins() -> None:
    """Import built-ins and every module dropped into ``simlab.plugins``."""
    for package_name in ("simlab.builtin", "simlab.plugins"):
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            continue
        for module in pkgutil.iter_modules(package.__path__, package_name + "."):
            importlib.import_module(module.name)
    # Register the TorchPolicy adapter without importing torch at module load
    # (torch is imported lazily inside TorchPolicy methods).
    try:
        importlib.import_module("simlab.training.torch_policy")
    except ModuleNotFoundError:
        pass
