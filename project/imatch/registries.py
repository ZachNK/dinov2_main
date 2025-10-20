# project/imatch/registries.py
"""
Weight/alias registry helpers shared across CLI entry points.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# alias -> checkpoint filename
WEIGHT_FILES = {
    # 1) ViT_LVD-1689M
    "vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vith16+": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vits16+": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",

    # 2) ConvNeXT_LVD-1689M
    "cxBase": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "cxLarge": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    "cxSmall": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "cxTiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",

    # 3) ViT_SAT-493M
    "vit7b16sat": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    "vitl16sat": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}

# alias -> torch.hub entry name
HUB_BY_ALIAS = {
    "vit7b16": "dinov3_vit7b16",
    "vitb16": "dinov3_vitb16",
    "vith16+": "dinov3_vith16plus",
    "vitl16": "dinov3_vitl16",
    "vits16": "dinov3_vits16",
    "vits16+": "dinov3_vits16plus",

    "cxBase": "convnext_base",
    "cxLarge": "convnext_large",
    "cxSmall": "convnext_small",
    "cxTiny": "convnext_tiny",

    "vit7b16sat": "dinov3_vit7b16",
    "vitl16sat": "dinov3_vitl16",
}

# logical groups exposed to CLI
WEIGHT_GROUPS = {
    "ViT_LVD1689M": ["vit7b16", "vitb16", "vith16+", "vitl16", "vits16", "vits16+"],
    "ConvNeXT_LVD1689M": ["cxBase", "cxLarge", "cxSmall", "cxTiny"],
    "ViT_SAT493M": ["vit7b16sat", "vitl16sat"],
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
        hub_name = HUB_BY_ALIAS.get(alias, "dinov3_vitl16")
        results.append((alias, hub_name, resolved_path))
    return results

