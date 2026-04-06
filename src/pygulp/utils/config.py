import os
from pathlib import Path
import yaml


def find_project_root():
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def load_config(config_path=None):
    """Load configuration from base.yml and resolve paths."""
    root = find_project_root()

    if config_path is None:
        config_path = root / "configs" / "base.yml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((root / Path(value)).resolve())

    # Expand ~ in executable paths
    for key, value in cfg.get("executables", {}).items():
        cfg["executables"][key] = os.path.expanduser(value)

    return cfg


def setup_gulp_env(cfg=None):
    """Set GULP environment variables from config."""
    if cfg is None:
        cfg = load_config()

    gulp_exe = cfg["executables"]["gulp_dir"]
    gulp_lib = cfg["executables"]["gulp_lib"]

    os.environ["ASE_GULP_COMMAND"] = f"{gulp_exe} < PREFIX.gin > PREFIX.got"
    os.environ["GULP_LIB"] = gulp_lib
