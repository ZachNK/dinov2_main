# project/imatch/env.py
import os
from pathlib import Path
from typing import Optional


def getenv(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Environment loader with optional required enforcement.
    """
    value = os.getenv(key, default)
    if required and (value is None or str(value).strip() == ""):
        raise SystemExit(f"Missing env: {key}")
    return value


# Base directories (docker-compose.yml/.env inject absolute paths)
REPO_DIR = Path(getenv("REPO_DIR", required=True))
IMG_ROOT = Path(getenv("IMG_ROOT", required=True))

# Output roots (Windows host paths are mounted to /exports inside the container)
EMBED_ROOT = Path(getenv("EMBED_ROOT", "/exports/dinov3_embeds"))
MATCH_ROOT = Path(getenv("MATCH_ROOT", "/exports/dinov3_match"))
VIS_ROOT = Path(getenv("VIS_ROOT", "/exports/dinov3_vis"))

# Network guard: torch.hub remote downloads are disabled unless explicitly opted out
DINOV3_BLOCK_NET = getenv("DINOV3_BLOCK_NET", "1").strip() == "1"
