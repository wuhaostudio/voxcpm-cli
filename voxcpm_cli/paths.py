"""Path resolution for the VoxCPM CLI."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the CLI root directory."""
    env_root = os.environ.get("VOXCPM_CLI_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def original_model_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "original" / "VoxCPM2"


def openvino_model_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "openvino" / "VoxCPM2"


def output_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "output"


def cache_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "cache"


def resolve_output_path(value: str | None, root: Path | None = None) -> Path:
    base = output_dir(root).resolve()
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (project_root() / path).resolve()
        else:
            path = path.resolve()
    else:
        from datetime import datetime

        name = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        path = (base / name).resolve()

    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside {base}") from exc
    return path
