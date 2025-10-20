# project/imatch/paths.py
"""
Utility helpers for naming pair-match output files.
"""

from pathlib import Path

# TODO: expose via env if needed. Current convention is fixed inside container.
PAIR_MATCH_ROOT = Path("/exports/pair_match")


def split_key(key: str) -> tuple[str, str]:
    alt, frame = key.split(".")
    return alt, frame


def out_dir_for_pair(weight_alias: str, key_a: str) -> Path:
    alt, frame = split_key(key_a)
    return PAIR_MATCH_ROOT / f"{weight_alias}_{alt}_{frame}"


def out_name_for_pair(weight_alias: str, key_a: str, key_b: str) -> str:
    return f"{weight_alias}_{key_a}_{key_b}"

