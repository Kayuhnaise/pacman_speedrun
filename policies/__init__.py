# policies/__init__.py
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

_modules: dict[str, ModuleType] = {}


def _load_submodules() -> None:
    pkg_dir = Path(__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(pkg_dir)]):
        if name == "__pycache__":
            continue
        full_name = f"policies.{name}"
        try:
            mod = importlib.import_module(full_name)
            _modules[name] = mod
        except Exception:
            continue


def get_attribute(name: str) -> Optional[Callable]:
    if not _modules:
        _load_submodules()

    if not name:
        return None

    if "." in name:
        mod_name, attr = name.split(".", 1)
        try:
            mod = importlib.import_module(f"policies.{mod_name}")
            return getattr(mod, attr)
        except Exception:
            return None

    for mod in _modules.values():
        if hasattr(mod, name):
            attr = getattr(mod, name)
            if callable(attr):
                return attr

    mod = _modules.get(name)
    if mod:
        if hasattr(mod, "policy") and callable(getattr(mod, "policy")):
            return getattr(mod, "policy")
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return getattr(mod, name)

    return None