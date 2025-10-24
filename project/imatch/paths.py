# project/imatch/paths.py
"""
Utility helpers for naming pair-match output files.
"""

from pathlib import Path

from imatch.env import MATCH_ROOT


def split_key(key: str) -> tuple[str, str]:
    alt, frame = key.split(".")
    return alt, frame


def match_root() -> Path:
    """
    Expose the configured match root (JSON destination) for callers that need the raw path.
    """
    return MATCH_ROOT


def out_dir_for_pair(weight_alias: str, key_a: str) -> Path:
    alt, frame = split_key(key_a)
    return MATCH_ROOT / f"{weight_alias}_{alt}_{frame}"


def out_name_for_pair(weight_alias: str, key_a: str, key_b: str) -> str:
    return f"{weight_alias}_{key_a}_{key_b}"
