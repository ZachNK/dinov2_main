# imatch/tfms.py
import math
import torch
from torchvision import transforms
def build_transform(
    ### 이미지 전처리(transform) 파이프라인을 구축한다.
    # patch_size: 모델의 패치 크기 (예: 16, 14 등)
    # patch_multiple: 입력 이미지 크기를 패치 크기의 배수로 조정 (기본값: 16)
    # interpolation: 리사이즈 시 사용할 보간법 (기본값: "bicubic")
    # normalize: 이미지 정규화 여부 (기본값: True)  
    patch_size: int, # 모델 패치 크기 (예: 16, 14 등)
    patch_multiple: int = 16, # 입력 이미지 크기를 패치 크기의 배수로 조정 (16, 14 등)
    interpolation: str = "bicubic", # 보간법 bicubic 사용 (bilinear, nearest 등도 가능)
    normalize: bool = True,        
):
    ### 입력 이미지 크기를 패치 크기의 배수로 설정
    # target_size: 입력 이미지의 최종 크기 = 모델 패치 크기 X 패치 크기의 배수
    target_size = patch_size * patch_multiple
    # transform_steps: 전처리 단계 리스트 초기화
    transform_steps = [
        # 이미지를 float32 타입으로 변환
        transforms.ConvertImageDtype(torch.float32),
        # 이미지를 target_size x target_size 크기로 리사이즈
        transforms.Resize(
            # 리사이즈 대상 크기
            (target_size, target_size),
            # 리사이즈시 보간법 적용
            interpolation=getattr(transforms.InterpolationMode, interpolation.upper()),
            # 안티앨리어싱 적용 여부
            antialias=True,
        ),
    ]
    ### 이미지 정규화 단계 추가
    if normalize:
        ### 표준 ImageNet Normalization 값 사용
        transform_steps.append(
            # 채널별 평균 및 표준편차로 정규화
            transforms.Normalize(
                # 채널별 평균값 (R, G, B 순서)
                mean=[0.485, 0.456, 0.406],
                # 채널별 표준편차값 (R, G, B 순서)
                std=[0.229, 0.224, 0.225],
            )
        )
    # 최종 전처리 파이프라인 반환
    return transforms.Compose(transform_steps)
