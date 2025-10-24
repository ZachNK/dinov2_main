"""
DINOv3 torch.hub 모델을 직접 로드해서 단일 이미지를 추론하는 간단한 스크립트.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from imatch.features import extract_global_feature
from imatch.io_images import load_image_tensor
from imatch.tfms import build_transform


REPO_DIR = Path("/workspace/dinov3")
CKPT_PATH = Path("/opt/weights/01_ViT_LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
IMAGE_PATH = Path("/opt/datasets/250912143954_450/250912143954_450_0001.jpg")
HUB_ENTRY = "dinov3_vitl16"
IMAGE_SIZE = 336


def load_dinov3_model() -> torch.nn.Module:
    model = torch.hub.load(
        str(REPO_DIR),
        HUB_ENTRY,
        source="local",
        trust_repo=True,
        pretrained=False,
    )
    try:
        state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(CKPT_PATH), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[ckpt][warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[ckpt][warn] unexpected keys: {len(unexpected)}")
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov3_model().to(device).eval()

    img_tensor = load_image_tensor(IMAGE_PATH)
    transform = build_transform(IMAGE_SIZE)
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)

    with torch.inference_mode():
        pooled_vec = extract_global_feature(model, input_tensor, str(device))

    pooled_vec = pooled_vec.detach().cpu()
    print("Pooled feature shape:", tuple(pooled_vec.shape))
    print("Pooled feature (first 10 elements):", pooled_vec[:10].tolist())

    export_dir = Path("/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    npy_path = export_dir / "pooled_feature.npy"
    csv_path = export_dir / "pooled_feature.csv"

    pooled_arr = pooled_vec.numpy()
    np.save(npy_path, pooled_arr)
    np.savetxt(csv_path, pooled_arr[None, :], delimiter=",")
    print(f"[saved] numpy array -> {npy_path}")
    print(f"[saved] csv row     -> {csv_path}")


if __name__ == "__main__":
    main()
