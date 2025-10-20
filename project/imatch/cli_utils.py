# project/imatch/cli_utils.py
"""
Helpers for building CLI argument parsers.
"""

import argparse
from typing import Callable


def bounded_float(low: float, high: float) -> Callable[[str], float]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """
    def _validate(raw: str) -> float:
        value = float(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value
    return _validate


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """
    def _validate(raw: str) -> int:
        value = int(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value
    return _validate

