# project/imatch/registries.py
"""
Weight/alias registry helpers shared across CLI entry points.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# alias -> checkpoint filename
WEIGHT_FILES = {
    # 01_weights
    "vitb14": "dinov2_vitb14_pretrain.pth",
    "vits14": "dinov2_vits14_pretrain.pth",
    "vitg14": "dinov2_vitg14_pretrain.pth",
    "vitl14": "dinov2_vitl14_pretrain.pth",

    # 02_with_registers
    "vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
    "vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
    "vitg14_reg": "dinov2_vitg14_reg4_pretrain.pth",
    "vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
}

# alias -> torch.hub entry name
HUB_BY_ALIAS = {
    "vitb14": "dinov2_vitb14",
    "vits14": "dinov2_vits14",
    "vitg14": "dinov2_vitg14",
    "vitl14": "dinov2_vitl14", 
    "vits14_reg": "dinov2_vits14_reg4",
    "vitb14_reg": "dinov2_vitb14_reg4",
    "vitg14_reg": "dinov2_vitg14_reg4",
    "vitl14_reg": "dinov2_vitl14_reg4",
}

# logical groups exposed to CLI
WEIGHT_GROUPS = {
    "01_weights": ["vitb14", "vits14", "vitg14", "vitl14"],
    "02_with_registers": ["vits14_reg", "vitb14_reg", "vitg14_reg", "vitl14_reg",],
}


def resolve_weight_paths(
    aliases: Sequence[str],
    search_roots: Iterable[Path],
) -> List[Tuple[str, str, Path]]:
    """
    Resolve a list of weight aliases to concrete checkpoint paths.

    Returns triplets of (alias, torch.hub entry, path).
    """
    results: List[Tuple[str, str, Path]] = []
    for alias in aliases:
        if alias not in WEIGHT_FILES:
            raise SystemExit(f"Unknown weight alias: {alias}")
        filename = WEIGHT_FILES[alias]
        resolved_path: Path | None = None
        for root in search_roots:
            candidate = Path(root) / filename
            if candidate.is_file():
                resolved_path = candidate
                break
        if resolved_path is None:
            raise SystemExit(f"[ckpt] not found for {alias}: {filename}")
        hub_name = HUB_BY_ALIAS.get(alias, "dinov2_vitl14")
        results.append((alias, hub_name, resolved_path))
    return results

