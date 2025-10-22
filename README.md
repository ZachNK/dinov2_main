# DINOv3 Image Matching (Docker)

Docker Desktop 위에서 DINOv3 기반 이미지 매칭과 시각화를 수행하기 위한 프로젝트.  
컨테이너 안에서는 1:1 매칭을 수행하도록 구성되어 있으며, 결과(JSON/PNG)는 호스트의 지정된 디렉터리에 저장.

---

## 0) 요구 사항

- **Windows 11** + **Docker Desktop (v.4.46.0 이상)**  
  - Docker Desktop 환경에서 동작.  
  - Docker Desktop Settings → Resources → File Sharing 에서 프로젝트/데이터 폴더가 공유되어 있는지 확인.
- **NVIDIA GPU & 최신 드라이버** (CUDA 12.x 호환)
- **NVIDIA Container Toolkit** (Docker Desktop 설치 시 자동 포함)
- 권장 체크 명령
  ```powershell
  docker --version
  nvidia-smi
  ```

---

## 1) 저장소 구조 & 필수 리소스

```
dinov3_docker/
├─ project/
│  ├─ imatch/           # 라이브러리 모듈
│  ├─ run.py            # 매칭 실행 엔트리
│  └─ visualize.py      # 시각화 엔트리
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .env.example
└─ README.md
```

필수 리소스
- **facebookresearch/dinov3** 저장소 (코드 참조용)  
  예시 위치: `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_main`
- **사전 학습 가중치(.pth)**  
  예시 위치: `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_weights`
- **매칭 대상 이미지 데이터셋**  
  예시 위치: `D:\GoogleDrive\KNK_Lab\_Datasets\shinsung_data`
- **결과 저장 디렉터리**  
  예시 위치: `D:\GoogleDrive\KNK_Lab\Exports`

### 1-1) DINOv3 원본 저장

작업하고자 하는 디렉토리에 먼저 접근하여 본 프로젝트를 `dinov3_docker` 하위 경로에 clone한다. 

```powershell
git clone https://github.com/ZachNK/ImgMatching_DINOv3.git .\dinov3_docker
```

### 1-2) DINOv3 원본 저장

(작업할 디렉토리: `C:\a\b\c\d` 라고 가정)\
작업할 경로 (`C:\a\b\c\d`)에 DINOv3 원본을 저장한다.

```powershell
git clone https://github.com/facebookresearch/dinov3.git .\dinov3_main
```

### 1-3) 백본 백본 준비 

그리고 상위 경로에 `dinov3_weights` 라는 디렉토리로 백본 데이터를 준비 한다.\
https://github.com/facebookresearch/dinov3에 게시된 가중치를 `dinov3_weights`이라는 폴더를 생성하고 바로 저장한다.\
만약 방금 DINOv3원본을 clone한 디렉토리가 `C:\a\b\c\d\dinov3_main`이라 가정한다면: 

```powershell
# dinov3_weights 디렉토리 생성. 해당 경로에 ViT-S/16 distilled, ConvNeXt Tiny, ... , ViT-7B/16 를 저장
New-Item -ItemType Directory -Path C:\a\b\c\d\dinov3_weights -ErrorAction SilentlyContinue
```

```powershell
# dinov3_weights에 디렉토리 추가 생성
New-Item -ItemType Directory -Path C:\a\b\c\d\dinov3_weights\01_ViT_LVD-1689M -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path C:\a\b\c\d\dinov3_weights\02_ConvNeXT_LVD-1689M -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path C:\a\b\c\d\dinov3_weights\03_ViT_SAT-493M -ErrorAction SilentlyContinue
```

```powershell
# dinov3_weights 디렉토리에 저장된 .pth 파일들 데이터셋별로 정리

# dinov3_weights\01_ViT_LVD-1689M에 파일 이동 (ViT-S/16 distilled 이동할 때)
Move-Item -Path C:\a\b\c\d\dinov3_vits16_pretrain_lvd1689m-08c60483.pth -Destination C:\a\b\c\d\dinov3_weights\01_ViT_LVD-1689M

# ... 나머지 ViT-S+/16 distilled, ViT-B/16 distilled 등 .pth파일 이동

# dinov3_weights\02_ConvNeXT_LVD-1689M에 파일 이동 (ConvNeXt Tiny 이동할 때)
Move-Item -Path C:\a\b\c\d\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth -Destination C:\a\b\c\d\dinov3_weights\01_ViT_LVD-1689M

# ... 나머지 ConvNeXt Small, ConvNeXt Base 등 .pth파일 이동

# dinov3_weights\03_ViT_SAT-493M에 파일 이동 (ViT-L/16 distilled 이동할 때)
Move-Item -Path C:\a\b\c\d\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth -Destination C:\a\b\c\d\dinov3_weights\01_ViT_LVD-1689M

# ... 나머지 dinov3_vit7b16_pretrain_sat493m-a6675841.pth .pth파일 이동
```




### 2-4) 데이터셋 준비

마찬가지로 데이터셋도 추가 디렉토리를 생성한다.

```powershell
# dinov3_data 디렉토리 생성.
New-Item -ItemType Directory -Path C:\a\b\c\dinov3_data
```


### 2-5) 결과 저장 생성


---



---

## 2) `.env` 설정

`cp .env.example .env` 후 자신의 환경에 맞게 수정. 모든 경로는 **Windows 경로**로 작성.

| 변수 | 설명 | 예시 (Windows) |
| --- | --- | --- |
| `PROJECT_HOST` | `project/` 폴더 실경로 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_docker\project` |
| `CODE_HOST` | dinov3 원본 리포지터리 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_main` |
| `WEIGHTS_HOST` | `.pth` 가중치 루트 | `D:\GoogleDrive\KNK_Lab\_Projects\dinov3_weights` |
| `DATASET_HOST` | 이미지 데이터셋 루트 | `D:\GoogleDrive\KNK_Lab\_Datasets\shinsung_data` |
| `EXPORT_HOST` | JSON/PNG 결과 저장 루트 | `D:\GoogleDrive\KNK_Lab\Exports` |
| `REPO_DIR` | 컨테이너 내부 dinov3 마운트 위치 | `/workspace/dinov3` |
| `IMG_ROOT` | 컨테이너 내부 데이터셋 위치 | `/opt/datasets` |
| `EXPORT_DIR` | 컨테이너 내부 임베딩/결과 루트 | `/exports/dinov3_embeds` |
| `PAIR_VIZ_DIR` | 컨테이너 내부 시각화 결과 루트 | `/exports/pair_viz` |
| `IMATCH_VIZ_ROOT` | 매칭 JSON 기본 위치 | `/exports/pair_match` |
| `IMATCH_VIZ_OUT` | PNG 출력 위치 | `/exports/pair_viz` |
| `TZ` | 컨테이너 시간대 | `Asia/Seoul` |
| `DINOV3_BLOCK_NET` | torch.hub 다운로드 차단 (0/1) | `1` |


---

## 3) Docker 이미지 빌드 & 컨테이너 실행

```powershell
docker compose build        # Dockerfile 변경 시 재빌드
docker compose up -d        # 컨테이너 백그라운드 실행
docker compose ps           # 상태 확인
```

변경 사항 적용 또는 `.env`를 수정한 뒤에는 `docker compose up -d --force-recreate`로 재생성.  
GPU가 인식되는지 확인:
```powershell
docker compose exec pair nvidia-smi
```

---

## 4) 매칭 실행 (`run`)

컨테이너 래퍼 명령은 `run` 입니다.

```powershell
docker compose exec pair run --weights vitl16 -a 400.0200 -b 200.0200
```

- `-a`, `-b`: ALT.FRAME 형식 (예: `400.0200`)  
  지정하지 않으면 모든 조합을 순회.
- `--weights`: 사용 모델 alias (여러 개 지정 가능)  
  `--group`, `--all-weights` 옵션도 지원.

  | backbone | flag |
  | -------- | ---- |
  | `ViT-S/16 distilled` | `vits16` |    
  | `ViT-S+/16 distilled` |`vits16+` |   
  | `ViT-B/16 distilled` | `vitb16` |
  | `ViT-L/16 distilled` | `vitl16` | 
  | `ViT-H+/16 distilled` | `vith16+` |
  | `ViT-7B/16` | `vit7b16` |
  | `ConvNeXt Tiny` | `cxTiny` |  
  | `ConvNeXt Small` | `cxSmall` |  
  | `ConvNeXt Base` | `cxBase` |   
  | `ConvNeXt Large` | `cxLarge` |  
  | `ViT-L/16 distilled` | `vitl16sat` | 
  | `ViT-7B/16` | `vit7b16sat` |


- 주요 튜닝 파라미터
  | 옵션 | 기본값 | 설명 |
  | --- | --- | --- |
  | `--image-size` | 336 | 입력 해상도 |
  | `--max-features` | 1000 | 패치 토큰 최대 개수 (균등 샘플링) |
  | `--match-th` | 0.1 | 유사도 절대 임계값 |
  | `--keypoint-th` | 0.015 | 토큰 L2 임계값 |
  | `--line-th` | 0.2 | 최고 유사도 대비 상대 임계값 |

- 결과 JSON은 `/exports/pair_match/<weight>_<ALT>_<FRAME>/…` 에 저장.

---

## 5) 시각화 (`vis`)

```powershell
# 대화형 선택 
docker compose exec pair vis


주요 옵션
| 옵션 | 기본값 (환경변수) | 설명 |
| --- | --- | --- |
| `--root` | `/exports/pair_match` (`IMATCH_VIZ_ROOT`) | JSON 루트 |
| `--out` | `/exports/pair_viz` (`IMATCH_VIZ_OUT`) | PNG 출력 루트 |
| `--ransac` | `homography` | `off/affine/homography` |
| `--reproj-th` | 8.0 | 투영 오차 임계값 |
| `--confidence` | 0.9999 | RANSAC 신뢰도 |
| `--max-lines` | 1000 | 그릴 매칭 수 (0이면 미표시) |
| `--draw-points` | OFF | 점 표시 여부 |

실행 결과는 호스트 `EXPORT_HOST\pair_viz\…`에서 확인.

---

## 6) 결과 확인 & 경로 정리

- JSON: `EXPORT_HOST\pair_match\<weight>_<ALT>_<FRAME>\*.json`
- PNG: `EXPORT_HOST\pair_viz\<weight>_<ALT>_<FRAME>\*.png`
- JSON 내용
  - `meta`: 실행 환경 정보
  - `advanced_settings`: 사용한 매칭/필터 파라미터 (`matching_mode`는 현재 `mutual_knn_k1_unique`)
  - `patch`: 선택된 패치 정보 (`idx_a`, `idx_b`, `similarities` 등)

필요 시 결과 폴더를 탐색기에서 바로 열어 확인.

---

## 7) 트러블슈팅

| 증상 | 확인 사항 & 해결 팁 |
| --- | --- |
| Docker 명령 실패 / 권한 오류 | Docker Desktop 재시작, 관리자 PowerShell에서 실행 |
| 컨테이너에서 GPU 미노출 | `docker compose exec pair nvidia-smi` 확인 → NVIDIA 드라이버/NVIDIA Container Toolkit 재설치 |
| 볼륨 마운트 실패 | Docker Desktop Settings → Resources → File Sharing 에서 각 드라이브 허용 여부 확인 |
| 매칭 JSON이 생성되지 않음 | `pairs_to_run=0` 인 경우 ALT.FRAME 조합이 존재하지 않는 것 → 데이터셋 이름/정규식 확인 |
| 1:1 매칭이 맞지 않는 것처럼 보임 | `run` 내부에서 자동으로 1:1을 강제함. PNG 상에서 선이 적게 보인다면 RANSAC 필터를 완화하거나 `vis --ransac off`로 검증 |
| 기타 로그 | `docker compose logs -f pair` 로 컨테이너 로그 확인 |

---

필요한 경우 `docker compose exec pair bash`로 컨테이너 내부에 진입하여 추가 디버깅을 진행 가능.  

