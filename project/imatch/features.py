# project/imatch/features.py
import torch
from typing import Optional, Tuple


@torch.no_grad()
def extract_global_feature(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """
    모델 출력에서 글로벌 특징을 추출한다.
    """
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if isinstance(out, dict):
        if "x" in out and isinstance(out["x"], torch.Tensor) and out["x"].ndim == 3:
            feat = out["x"].mean(dim=1)
        else:
            for k in ("feat", "features", "pooled", "pooler_output"):
                if k in out and isinstance(out[k], torch.Tensor):
                    feat = out[k]
                    break
            else:
                feat = [v for v in out.values() if torch.is_tensor(v)][-1]
    else:
        feat = out

    if feat.ndim == 3:
        feat = feat.mean(dim=1)
    if feat.ndim == 4:
        feat = feat.mean(dim=(2, 3))
    return feat.squeeze(0)


@torch.no_grad()
def extract_patch_tokens(model: torch.nn.Module, x: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """
    패치 토큰(CLS 제외)을 추출한다.
    """
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if isinstance(out, dict):
        if "x_norm_patchtokens" in out and torch.is_tensor(out["x_norm_patchtokens"]):
            v = out["x_norm_patchtokens"]
            if v.ndim == 3:
                return v.squeeze(0).contiguous()
        for k in ["patch_tokens", "tokens_patch", "features_patch"]:
            v = out.get(k, None)
            if torch.is_tensor(v) and v.ndim == 3:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        for v in out.values():
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        return None

    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        return None

    if torch.is_tensor(out) and out.ndim == 3 and out.shape[1] > 16:
        return (out[:, 1:, :].squeeze(0) if out.shape[1] > 1 else out.squeeze(0)).contiguous()

    return None


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    코사인 유사도를 계산한다.
    """
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())


def apply_keypoint_threshold(
    tokens: torch.Tensor,
    idx_map: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    토큰 L2-노름 기반의 임계값 필터링을 수행한다. 모든 토큰이 걸러지는 경우
    최고 점수 토큰을 하나 남겨 매칭 단계가 비어 있지 않도록 보장한다.
    """
    if tokens.numel() == 0:
        return tokens, idx_map

    scores = torch.linalg.norm(tokens, dim=1)
    min_s = scores.min()
    max_s = scores.max()
    if (max_s - min_s).abs() < 1e-6:
        normalized = torch.ones_like(scores)
    else:
        normalized = (scores - min_s) / (max_s - min_s + 1e-6)

    mask = normalized >= threshold
    if not torch.any(mask):
        top_idx = torch.argmax(normalized)
        mask[top_idx] = True

    keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    filtered_tokens = tokens.index_select(0, keep_idx)
    filtered_idx_map = idx_map.index_select(0, keep_idx)
    return filtered_tokens, filtered_idx_map

