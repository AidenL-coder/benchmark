"""Configuration loading.

Brief section 11: "config-driven (YAML/JSON) experiments; one command reproduces
one experiment", and "no magic constants in code -- everything in a config file".

Every config is resolved into a plain dict *and* fingerprinted, so a result can
be traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Config", "load_config", "ConfigError"]


class ConfigError(ValueError):
    """A configuration is missing a required key or holds an invalid value."""


class _Missing:
    """Sentinel distinguishing "no default given" from a default of ``None``."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Fetch a dotted path, e.g. `config.get("frontier.n_max")`."""
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if isinstance(default, _Missing):
                    raise ConfigError(
                        f"missing required config key {path!r}"
                        + (f" in {self.source}" if self.source else "")
                    )
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        value = self.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"config section {name!r} must be a mapping")
        return value

    def fingerprint(self) -> str:
        """Stable hash of the resolved config, recorded with every result."""
        payload = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict:
        return json.loads(json.dumps(self.data, default=str))

    def with_overrides(self, overrides: dict) -> "Config":
        return Config(data=_deep_merge(self.data, overrides), source=self.source)


def load_config(path: Path | str, overrides: dict | None = None) -> Config:
    """Load a YAML/JSON config, applying `extends` inheritance then overrides."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")

    raw = _read_structured(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")

    # `extends` lets an experiment inherit a profile (e.g. the Colab profile)
    # without duplicating it, keeping the diff between runs legible.
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path).data
        raw = _deep_merge(base, raw)

    config = Config(data=raw, source=path)
    if overrides:
        config = config.with_overrides(overrides)
    return config


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("PyYAML is required to read YAML configs") from exc
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ConfigError(f"unsupported config format: {path.suffix}")


def parse_cli_overrides(pairs: list[str]) -> dict:
    """Turn `--set frontier.n_max=1000` style pairs into a nested dict."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override must be key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
