"""
visualize.py
- runCLI.py 가 생성한 JSON을 읽어 좌/우 이미지 매칭을 PNG로 저장
- 입력: --root (기본 /exports/pair_match) 내 JSON
- 출력: $PAIR_VIZ_DIR/<weight>_<Aalt>_<Aframe>/<same_base>.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from imatch.env import PAIR_VIZ_DIR

PAIR_MATCH_ROOT = Path("/exports/pair_match")  # JSON 기본 루트


def _discover_homography_methods() -> Tuple[dict, str]:
    """
    OpenCV가 제공하는 homography RANSAC flag 들을 탐색.
    반환: (name->flag mapping, default name)
    """
    candidates = {
        "cv2_ransac": ["RANSAC"],
        "cv2_usac_default": ["USAC_DEFAULT"],
        "cv2_usac_parallel": ["USAC_PARALLEL"],
        "cv2_usac_accurate": ["USAC_ACCURATE"],
        "cv2_usac_fast": ["USAC_FAST"],
        "cv2_usac_prosac": ["USAC_PROSAC"],
        "cv2_usac_magsac": ["USAC_MAGSAC"],
        "cv2_usac_fm_8pts": ["USAC_FM_8PTS"],
        "poselib": ["USAC_POSE", "USAC_POSELIB", "USAC_RELATIVE_POSE"],
    }
    mapping = {}
    for name, attr_names in candidates.items():
        for attr in attr_names:
            flag = getattr(cv2, attr, None)
            if isinstance(flag, int):
                mapping[name] = flag
                break
    default_name = "cv2_usac_magsac" if "cv2_usac_magsac" in mapping else "cv2_ransac"
    if default_name not in mapping and mapping:
        default_name = next(iter(mapping.keys()))
    return mapping, default_name


HOMOGRAPHY_METHODS, HOMOGRAPHY_DEFAULT = _discover_homography_methods()


# ---------- 유틸 ----------
def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def clamp(value, low, high):
    return max(low, min(high, value))


def bounded_int(low: int, high: int):
    def _check(raw: str) -> int:
        val = int(raw)
        if not (low <= val <= high):
            raise argparse.ArgumentTypeError(f"value {val} not in [{low}, {high}]")
        return val
    return _check


def bounded_float(low: float, high: float):
    def _check(raw: str) -> float:
        val = float(raw)
        if not (low <= val <= high):
            raise argparse.ArgumentTypeError(f"value {val} not in [{low}, {high}]")
        return val
    return _check


def list_focus_candidates(root: Path) -> List[str]:
    candidates: List[str] = []
    if not root.exists():
        return candidates
    for entry in sorted(root.iterdir()):
        rel = entry.name
        if entry.is_dir():
            candidates.append(f"{rel}/")
        elif entry.is_file() and entry.suffix.lower() == ".json":
            candidates.append(rel)
    return candidates


def prompt_focus(root: Path) -> List[str]:
    candidates = list_focus_candidates(root)
    if not sys.stdin.isatty():
        raise SystemExit("No focus specified and interactive selection unavailable (non-tty).")
    if not candidates:
        print("[info] focus candidates가 없어 전체 JSON을 시각화합니다.")
        return ["**/*.json"]

    print("[focus] 선택 가능한 항목:")
    for idx, rel in enumerate(candidates, start=1):
        print(f"  {idx:2d}. {rel}")
    print("  a. all (전체)")

    while True:
        choice = input("시각화할 항목 번호 또는 경로(콤마 구분)를 입력하세요: ").strip()
        if not choice:
            print("  입력이 비어 있습니다. 다시 입력하세요.")
            continue
        if choice.lower() in ("a", "all"):
            return ["**/*.json"]

        selected: List[str] = []
        valid = True
        for token in choice.split(","):
            tok = token.strip()
            if not tok:
                continue
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(candidates):
                    selected.append(candidates[idx].rstrip("/"))
                else:
                    print(f"  잘못된 번호: {tok}")
                    valid = False
                    break
            else:
                selected.append(tok.rstrip("/"))
        if valid and selected:
            return selected
        print("  다시 입력하세요.")


def add_json_path(acc: List[Path], seen: Set[Path], path: Path) -> None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    acc.append(resolved)


def collect_jsons(root: Path, selections: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()
    for sel in selections:
        if not sel:
            continue
        entry = Path(sel)
        candidates: List[Path] = []

        if entry.is_absolute():
            candidates = [entry]
        else:
            entry_rel = root / entry
            if entry_rel.exists():
                candidates = [entry_rel]
            else:
                candidates = list((root).glob(sel))
                if not candidates and not sel.startswith("**/"):
                    candidates = list((root).glob(f"**/{sel}"))

        if not candidates:
            print(f"[warn] 선택한 항목에 해당하는 JSON을 찾지 못했습니다: {sel}")
            continue

        for cand in candidates:
            if cand.is_dir():
                for jp in sorted(cand.rglob("*.json")):
                    add_json_path(files, seen, jp)
            else:
                add_json_path(files, seen, cand)

    return sorted(files)


# ---------- 좌표 복원 유틸 ----------
def best_rect_grid(n: int) -> Tuple[int, int]:
    g = int(round(np.sqrt(n)))
    if g * g == n:
        return g, g
    best = (1, n)
    best_gap = n
    for h in range(1, g + 2):
        w = int(np.ceil(n / h))
        if h * w >= n:
            gap = abs(h - w)
            if gap < best_gap:
                best_gap = gap
                best = (h, w)
    return best


def idx_to_xy(idx: np.ndarray, H: int, W: int) -> np.ndarray:
    y = idx // W
    x = idx % W
    return np.stack([x, y], axis=1)


def grid_to_pixels(xy: np.ndarray, img_w: int, img_h: int, W: int, H: int) -> np.ndarray:
    px = (xy[:, 0] + 0.5) * (img_w / W)
    py = (xy[:, 1] + 0.5) * (img_h / H)
    return np.stack([px, py], axis=1)


# ---------- RANSAC ----------
def ransac_filter(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    method: str,
    homography_flag: int,
    reproj_thresh: float,
    confidence: float,
    max_iters: int,
) -> np.ndarray:
    N = min(ptsA.shape[0], ptsB.shape[0])
    if N == 0 or method == "off":
        return np.ones((N,), dtype=bool)

    if method == "homography" and N >= 4:
        H, mask = cv2.findHomography(
            ptsA, ptsB, homography_flag,
            ransacReprojThreshold=reproj_thresh,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)[:N]

    if method == "affine" and N >= 3:
        _, mask = cv2.estimateAffinePartial2D(
            ptsA, ptsB,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_thresh,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)[:N]

    return np.ones((N,), dtype=bool)


# ---------- 그리기 ----------
def hstack_images(imA: np.ndarray, imB: np.ndarray, pad: int = 8, color=(30, 30, 30)) -> Tuple[np.ndarray, int]:
    h = max(imA.shape[0], imB.shape[0])
    wA, wB = imA.shape[1], imB.shape[1]
    canvas = np.full((h, wA + pad + wB, 3), color, dtype=np.uint8)
    canvas[:imA.shape[0], :wA] = imA
    canvas[:imB.shape[0], wA + pad:wA + pad + wB] = imB
    return canvas, wA + pad


def draw_matches(
    canvas: np.ndarray,
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    xoffB: int,
    max_lines: int,
    linewidth: int,
    draw_points: bool,
    alpha: int,
    point_radius: int,
) -> None:
    if max_lines <= 0:
        return
    N = min(max_lines, ptsA.shape[0], ptsB.shape[0])
    if N <= 0:
        return

    draw_lines = linewidth > 0
    draw_pts = draw_points and point_radius != 0
    if not draw_lines and not draw_pts:
        return

    overlay = canvas.copy()
    color_line = (255, 200, 0)
    color_ptsA = (0, 220, 255)
    color_ptsB = (80, 255, 80)

    for i in range(N):
        x1, y1 = int(round(ptsA[i, 0])), int(round(ptsA[i, 1]))
        x2, y2 = int(round(ptsB[i, 0] + xoffB)), int(round(ptsB[i, 1]))
        if draw_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), color_line, linewidth, cv2.LINE_AA)
        if draw_pts:
            radius = point_radius if point_radius > 0 else max(1, linewidth + 1)
            cv2.circle(overlay, (x1, y1), radius, color_ptsA, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x2, y2), radius, color_ptsB, -1, cv2.LINE_AA)

    if alpha <= 0:
        canvas[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha / 255.0, canvas, 1.0 - alpha / 255.0, 0, canvas)


# ---------- 메인 ----------
def main():
    root_default = env_path("IMATCH_VIZ_ROOT", PAIR_MATCH_ROOT)
    out_default = env_path("IMATCH_VIZ_OUT", PAIR_VIZ_DIR)
    max_lines_default = clamp(env_int("IMATCH_VIZ_MAX_LINES", 1000), 0, 5000)
    linewidth_default = clamp(env_int("IMATCH_VIZ_LINEWIDTH", 3), 0, 20)
    alpha_default = clamp(env_int("IMATCH_VIZ_ALPHA", 180), 0, 255)
    point_radius_default = clamp(env_int("IMATCH_VIZ_POINT_RADIUS", -1), -1, 100)
    draw_points_default = env_bool("IMATCH_VIZ_DRAW_POINTS", False)
    ransac_env = os.environ.get("IMATCH_VIZ_RANSAC", "homography").lower()
    ransac_default = ransac_env if ransac_env in {"off", "affine", "homography"} else "homography"
    impl_env = os.environ.get("IMATCH_VIZ_RANSAC_IMPL", HOMOGRAPHY_DEFAULT)
    if impl_env not in HOMOGRAPHY_METHODS:
        impl_env = HOMOGRAPHY_DEFAULT
    reproj_default = clamp(env_float("IMATCH_VIZ_REPROJ_TH", 8.0), 0.0, 12.0)
    confidence_default = clamp(env_float("IMATCH_VIZ_CONFIDENCE", 0.9999), 0.0, 1.0)
    iters_default = clamp(env_int("IMATCH_VIZ_ITERS", 10000), 0, 100000)
    focus_env = os.environ.get("IMATCH_VIZ_FOCUS")
    focus_default = None
    if focus_env:
        focus_default = [part.strip() for part in focus_env.split(",") if part.strip()]

    ap = argparse.ArgumentParser(description="Visualize DINOv3 matches (from JSON)")
    ap.add_argument("--root", type=str, default=str(root_default),
                    help="매칭 JSON이 위치한 루트 (기본: IMATCH_VIZ_ROOT 또는 /exports/pair_match)")
    ap.add_argument("--focus", nargs="+", default=focus_default, metavar="PATH",
                    help="시각화 대상 디렉토리/JSON/패턴 (기본: 실행 시 선택)")
    ap.add_argument("--out", type=str, default=str(out_default),
                    help="결과 PNG 저장 루트 (기본: IMATCH_VIZ_OUT 또는 PAIR_VIZ_DIR)")
    ap.add_argument("--max-lines", type=bounded_int(0, 5000), default=max_lines_default,
                    help="그릴 최대 매칭 수 (0이면 미사용)")
    ap.add_argument("--linewidth", type=bounded_int(0, 20), default=linewidth_default,
                    help="매칭 선 두께 (0이면 선 미표시)")
    point_group = ap.add_mutually_exclusive_group()
    point_group.add_argument("--draw-points", dest="draw_points", action="store_true",
                             help="매칭 지점을 원으로 표시")
    point_group.add_argument("--no-points", dest="draw_points", action="store_false",
                             help="매칭 지점 표시 비활성화")
    ap.set_defaults(draw_points=draw_points_default)
    ap.add_argument("--point-radius", type=bounded_int(-1, 100), default=point_radius_default,
                    help="지점 반경 (-1: 자동, 0: 표시 안함)")
    ap.add_argument("--alpha", type=bounded_int(0, 255), default=alpha_default,
                    help="선/점 오버레이 알파값 (0-255)")
    ap.add_argument("--ransac", choices=["off", "affine", "homography"], default=ransac_default,
                    help="RANSAC 모드 (off/affine/homography)")
    ap.add_argument("--ransac-method", choices=sorted(HOMOGRAPHY_METHODS.keys()), default=impl_env,
                    help="homography RANSAC 플래그 (OpenCV 제공값만 표시)")
    ap.add_argument("--reproj-th", type=bounded_float(0.0, 12.0), default=reproj_default,
                    help="RANSAC 재투영 임계값 (0-12)")
    ap.add_argument("--confidence", type=bounded_float(0.0, 1.0), default=confidence_default,
                    help="RANSAC 신뢰도 (0-1)")
    ap.add_argument("--iters", type=bounded_int(0, 100000), default=iters_default,
                    help="RANSAC 반복 횟수 (0-100000)")
    args = ap.parse_args()

    pairs_root = Path(args.root).expanduser()
    if not pairs_root.exists():
        raise SystemExit(f"[error] root 디렉토리가 존재하지 않습니다: {pairs_root}")

    homography_flag = HOMOGRAPHY_METHODS.get(
        args.ransac_method,
        HOMOGRAPHY_METHODS.get(HOMOGRAPHY_DEFAULT, cv2.RANSAC),
    )

    focus_entries = args.focus
    if not focus_entries:
        focus_entries = prompt_focus(pairs_root)

    json_paths = collect_jsons(pairs_root, focus_entries)
    if not json_paths:
        print("[warn] 선택된 항목에서 JSON을 찾지 못했습니다.")
        return
    print(f"[focus] total_json={len(json_paths)}")

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))

        imgA = Path(data["image_a"])
        imgB = Path(data["image_b"])

        patch = data.get("patch", None)
        if not patch or not patch.get("idx_a") or not patch.get("idx_b"):
            print(f"[skip] no patch matches: {jp}")
            continue

        idx_a = np.array(patch["idx_a"], dtype=np.int64)
        idx_b = np.array(patch["idx_b"], dtype=np.int64)
        n_a = int(patch["n_a"])
        n_b = int(patch["n_b"])
        g_a = patch.get("grid_g_a")
        g_b = patch.get("grid_g_b")

        imA = np.array(Image.open(str(imgA)).convert("RGB"))[:, :, ::-1]
        imB = np.array(Image.open(str(imgB)).convert("RGB"))[:, :, ::-1]

        if g_a is not None:
            H_a = W_a = int(g_a)
        else:
            H_a, W_a = best_rect_grid(n_a)
        if g_b is not None:
            H_b = W_b = int(g_b)
        else:
            H_b, W_b = best_rect_grid(n_b)

        xy_a = idx_to_xy(idx_a, H_a, W_a)
        xy_b = idx_to_xy(idx_b, H_b, W_b)
        ptsA = grid_to_pixels(xy_a, imA.shape[1], imA.shape[0], W_a, H_a).astype(np.float32)
        ptsB = grid_to_pixels(xy_b, imB.shape[1], imB.shape[0], W_b, H_b).astype(np.float32)

        mask = ransac_filter(
            ptsA,
            ptsB,
            args.ransac,
            homography_flag,
            args.reproj_th,
            args.confidence,
            args.iters,
        )
        ptsA_in = ptsA[mask]
        ptsB_in = ptsB[mask]

        canvas, xoffB = hstack_images(imA, imB)
        draw_matches(
            canvas,
            ptsA_in,
            ptsB_in,
            xoffB,
            args.max_lines,
            args.linewidth,
            args.draw_points,
            args.alpha,
            args.point_radius,
        )

        try:
            rel_path = jp.relative_to(pairs_root)
        except ValueError:
            rel_path = Path(jp.name)
        out_path = out_root / rel_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        print(f"[keep] {out_path}")


if __name__ == "__main__":
    main()
