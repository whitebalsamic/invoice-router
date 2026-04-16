import logging
import os
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterable

import yaml

from .machine_profiles import _apply_processing_profile
from .schema import AppConfig
from .settings import Settings

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_ENV_VAR = "INVOICE_ROUTER_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"


def resolve_config_path(yaml_path: str | os.PathLike[str] | None = None) -> Path | Traversable:
    if yaml_path:
        return Path(yaml_path).expanduser()

    if env_path := os.environ.get(DEFAULT_CONFIG_ENV_VAR):
        return Path(env_path).expanduser()

    local_path = Path.cwd() / DEFAULT_CONFIG_NAME
    if local_path.exists():
        return local_path

    return files("invoice_router.config").joinpath("defaults.yaml")


def load_config(
    yaml_path: str | os.PathLike[str] | None = None,
    *,
    required_settings: Iterable[str] = (),
) -> tuple[Settings, AppConfig]:
    settings = Settings()  # type: ignore[call-arg]
    settings.require(*required_settings)

    config_path = resolve_config_path(yaml_path)
    if isinstance(config_path, Path):
        with config_path.open("r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)
    else:
        with config_path.open("r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)

    app_config = AppConfig(**yaml_dict)

    from . import get_paddle_ocr_status

    if app_config.ocr.table_engine == "paddle":
        paddle_ok, reason = get_paddle_ocr_status()
        if not paddle_ok:
            raise ValueError(
                "Configured OCR table engine 'paddle' is unavailable: "
                f"{reason}. Install a supported Paddle runtime or change "
                "ocr.table_engine to 'tesseract'."
            )

    _apply_processing_profile(app_config)

    return settings, app_config
