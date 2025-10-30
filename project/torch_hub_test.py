"""
DINOv2 torch.hub 모델을 직접 로드해서 단일 이미지를 추론하는 간단한 스크립트.
"""
from __future__ import annotations
import os
import warnings
import math
import torch
import numpy as np
from pathlib import Path as P
from imatch.features import extract_global_feature
from imatch.tfms import build_transform
from imatch.io_images import load_image_tensor
# 백본 모델, 체크포인트 경로, 이미지 경로, 허브 엔트리 이름, 이미지 크기 설정   
# ==== custom ====
IMG_DIR_NAME = "250912154506_300/250912154506_300_0010"
CKPT_PATH = P("/opt/weights/01_weights/dinov2_vitl14_pretrain.pth")
IMAGE_SIZE = 1024
# ==== custom ====

# 백본 모델, 체크포인트, 테스트 이미지 경로 설정
lst = IMG_DIR_NAME.split("/")[-1].split("_")
REPO_DIR = P("/workspace/dinov2")
IMAGE_PATH = P(f"/opt/datasets/{IMG_DIR_NAME}.jpg")
HUB_ENTRY = "_".join(os.path.splitext(os.path.basename(CKPT_PATH))[0].split("_")[:2])
FILE_NAME = f"global_feature_{HUB_ENTRY}_{lst[1]}_{lst[2]}"

# DINOv2 모델 로드 함수
def load_dinov2_model() -> torch.nn.Module: # torch.nn.Module 반환
    ### 로컬 저장소에서 DINOv2 vitl14 모델 로드
    # REPO_DIR.as_posix(): 로컬 경로 문자열
    # 'HUB_ENTRY': torch.hub에 등록된 모델 이름
    # source='local': 로컬 저장소에서 로드
    # trust_repo=False: 신뢰할 수 없는 저장소로 간주 (보안 경고 비활성화)
    model = torch.hub.load(REPO_DIR.as_posix(), HUB_ENTRY, source='local', trust_repo=False)
    ### 체크포인트 로드 및 모델 가중치 설정
    try:
        # 백본 모델을 GPU에 로드 시도, 가능하면 'cuda:0' 사용
        state = torch.load(CKPT_PATH.as_posix(), map_location="cuda:0")
    except TypeError:
        # 실패 시 CPU에 로드
        state = torch.load(CKPT_PATH.as_posix(), map_location="cpu")
    # 체크포인트가 'state_dict' 키를 포함하는 딕셔너리인 경우 해당 값으로 교체
    if isinstance(state, dict) and "state_dict" in state:
        # state_dict 형태로 가중치 추출
        state = state["state_dict"]
    # 'module.' 접두사가 붙은 키를 정리하여 모델에 맞게 조정
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # 모델에 정리된 가중치 로드, 엄격 모드 해제, 누락되거나 예기치 않은 키 경고 출력 (필요 시)
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    # 경고 출력
    if missing:
        print(f"[CKPT_PATH][warn] missing keys: {len(missing)}")
    # 예기치 않은 키가 있으면 출력
    if unexpected:
        print(f"[CKPT_PATH][warn] unexpected keys: {len(unexpected)}")
    # 모델 반환 
    return model
# 메인 함수: 모델 로드, 이미지 전처리, 특징 추출 및 저장
def main() -> None: # 반환값 없음
    # 불필요한 경고 무시 설정
    warnings.filterwarnings("ignore", message="xFormers is not available")
    # 장치 설정: CUDA 사용 가능 시 GPU, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DINOv2 모델 로드 및 평가 모드 설정
    model = load_dinov2_model().to(device).eval()
    # 이미지 로드 및 전처리
    img_tensor = load_image_tensor(IMAGE_PATH.as_posix())
    ### 전처리: 이미지 크기 조정 및 정규화
    patch_s = model.patch_embed.patch_size # 모델의 패치 크기 가져오기
    desired = IMAGE_SIZE
    
    patch_m = math.floor(desired / patch_s[0]) # 패치 크기의 배수 계산
    # build_transform 함수로 전처리기 빌드
    transform = build_transform(patch_size=patch_s[0], patch_multiple=patch_m, interpolation="bicubic", normalize=True) # vitl14 모델에 맞는 전처리 크기 설정
    
    print("img_tensor:", img_tensor.shape)

    # 전처리된 이미지 텐서를 배치 차원 추가 후 장치로 이동
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)
    
    ### 특징 추출: 전역 특징 벡터 계산
    with torch.inference_mode():
        # pooled_vec: 전역 특징 벡터
        pooled_vec = extract_global_feature(model, input_tensor, str(device))
    ### pooled_vec: 전역 특징 벡터를 CPU로 이동 후 분리
    pooled_vec = pooled_vec.detach().cpu()
    # 결과 출력: 형태 및 값
    print("Global feature shape:", tuple(pooled_vec.shape))
    # 결과 출력: 값 (리스트 형태)
    print("Global feature:", pooled_vec.tolist())
    ### 특징 백터(임베딩)을 numpy 배열 및 CSV로 저장
    export_dir = P("/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    # pooled_feature_dinov2.npy (npy 배열): 백본 정보 저장
    npy_path = export_dir / f"{FILE_NAME}.npy"
    # csv_path (CSV 파일): 백본 정보 저장
    csv_path = export_dir / f"{FILE_NAME}.csv"
    ### numpy 배열 및 CSV로 저장
    pooled_arr = pooled_vec.numpy()
    # npy_path: numpy 배열로 저장, pooled_arr: numpy 배열
    np.save(npy_path, pooled_arr)
    # csv_path: CSV 파일로 저장, pooled_arr[None, :]: 2D 배열로 변환
    np.savetxt(csv_path, pooled_arr[None, :], delimiter=",")
    ### 저장 완료 메시지 출력
    # npy 파일로 저장
    print(f"[saved] DINOv2 Test: numpy array -> {npy_path}")
    # 파일 완료로 저장
    print(f"[saved] DINOv2 Test: csv row     -> {csv_path}")
if __name__ == "__main__":
    main()