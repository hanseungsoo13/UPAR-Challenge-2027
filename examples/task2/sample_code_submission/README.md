# UPAR Task 2 ConvNeXt-B 베이스라인

> AttriVision CLIP baseline은 기존 코드를 건드리지 않고
> [`attrivision/`](attrivision/README.md)에 별도 구현되어 있다.

이 디렉터리는 UPAR Challenge 2027 Task 2인 **attribute-based person
retrieval**을 위한 학습, 로컬 검증, 체크포인트 관리 및 제출 추론 코드를
포함한다. 실행 진입점은 `run.py`이고, 실제 구현은 `upar/` 패키지에 기능별로
분리되어 있다.

## 1. 빠른 시작

저장소 루트에서 환경과 데이터셋을 준비한다.

```bash
conda env create -f environment.yml
conda activate rws-upar-challenge
python download_datasets.py
```

GPU 한 장을 지정해서 학습하려면 다음과 같이 실행한다. 아래 예시는 물리 GPU
4번만 노출하므로 프로그램 내부에서는 해당 GPU가 `cuda:0`이 된다.

```bash
CUDA_VISIBLE_DEVICES=4 python examples/task2/sample_code_submission/run.py \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/task2_baseline
```

기본 설정은 최대 100 epoch, batch size 64이며 validation mAP가 12회 연속
개선되지 않으면 조기 종료한다.

## 1.1 ConvNeXt ablation ladder

링크 baseline과의 차이를 하나씩 확인하려면 같은 seed와 데이터로 다음 preset을
순서대로 실행한다. 각 preset은 이전 output을 resume하지 않으며, 독립적인
`--output-dir`를 사용해야 한다.

```bash
# E1: official mADM으로 checkpoint 선택 및 평가
python examples/task2/sample_code_submission/run.py \
  --mode train_eval --ablation E1 \
  --output-dir outputs/ablation-E1

# E2: E1 + 256x128 입력, 기존 RandomResizedCrop 유지
python examples/task2/sample_code_submission/run.py \
  --mode train_eval --ablation E2 \
  --output-dir outputs/ablation-E2

# E3: E2 + Resize -> Pad(10) -> RandomCrop spatial policy
python examples/task2/sample_code_submission/run.py \
  --mode train_eval --ablation E3 \
  --output-dir outputs/ablation-E3
```

E1은 `selection_metric=madm`을 공통 anchor로 만든다. E2는 `256x128`만
변경하고, E3는 그 입력 크기에서 spatial crop policy만 바꾼다. E3에서도
AugMix는 유지하므로 crop policy의 효과를 분리해서 볼 수 있다. 각 run의
`metrics.csv`와 `train.log`에는 Rank-1, mAP, mADM이 함께 기록된다.

기존 checkpoint를 official metric으로 다시 확인할 때는 학습 없이 다음처럼
실행할 수 있다.

```bash
python examples/task2/sample_code_submission/run.py \
  --mode eval --ablation E1 \
  --checkpoint outputs/task2_baseline/best.pth
```

## 2. 필요한 데이터 구조

`--data-root`의 기본값은 저장소의 `data/` 디렉터리다. 최소한 다음 annotation
파일과 그 안에서 참조하는 이미지가 존재해야 한다.

```text
data/
├── annotations/task2/
│   ├── train/
│   │   ├── gt.csv
│   │   ├── ids.csv
│   │   └── queries.csv
│   └── val/
│       ├── gt.csv
│       ├── ids.csv
│       └── queries.csv
├── Market1501/
├── PA100k/
└── PETA/
```

`gt.csv`는 첫 열에 이미지 경로, 나머지 40개 열에 이진 attribute label을
가진다. `queries.csv`는 retrieval query, `ids.csv`는 각 gallery 이미지를
query 행 번호에 연결한다. `queries.csv`와 `ids.csv`가 없으면 `gt.csv`의
고유한 40-bit label 벡터로 query와 ID를 생성한다.

## 3. 모델과 학습 설정

- Backbone: ImageNet pretrained ConvNeXt-B
- 출력: 40개 attribute logit, 추론 시 sigmoid probability 사용
- 학습 augmentation: random resized crop, horizontal flip, AugMix
- 검증 전처리: resize, center crop, ImageNet normalization
- Loss: attribute 비율 가중 binary cross entropy + binary label smoothing
- Optimizer: AdamW
- Scheduler: validation mAP 기반 `ReduceLROnPlateau`
- EMA: 기본 활성화, 원본 모델과 EMA 모델을 모두 검증
- Retrieval distance: query와 attribute probability 사이의 L1 distance
- 최고 모델 선택 기준: validation mAP

주요 기본값은 다음과 같다.

| 설정 | 기본값 |
|---|---:|
| 최대 epoch | 100 |
| train batch size | 64 |
| eval batch size | 128 |
| learning rate | `1e-4` |
| weight decay | `5e-4` |
| dropout | `0.7` |
| label smoothing | `0.05` |
| EMA decay | `0.9998` |
| LR scheduler patience | 4회 검증 |
| early stopping patience | 12회 검증 |
| image size | 224 |

전체 옵션은 다음 명령으로 확인할 수 있다.

```bash
python examples/task2/sample_code_submission/run.py --help
```

## 4. 실행 모드

### 학습 후 자동 검증

```bash
python examples/task2/sample_code_submission/run.py \
  --mode train_eval \
  --output-dir outputs/task2_baseline
```

### 학습만 실행

```bash
python examples/task2/sample_code_submission/run.py \
  --mode train \
  --output-dir outputs/task2_baseline
```

### 저장된 모델 검증

```bash
python examples/task2/sample_code_submission/run.py \
  --mode eval \
  --checkpoint outputs/task2_baseline/best.pth
```

`--checkpoint`를 생략하면 `<output-dir>/best.pth`를 사용한다. 이전 코드에서
생성된 `<output-dir>/model_best.pth`만 존재하는 경우에는 해당 파일을 fallback으로
읽는다.

### 빠른 데이터·연산 검증

실제 학습 전에 annotation, loss, query, distance와 attribute 순서를 확인한다.

```bash
python examples/task2/sample_code_submission/run.py --mode smoke
```

ConvNeXt forward까지 확인하려면 `--smoke-model`을 추가한다. 소량의 데이터로
학습 파이프라인을 확인하려면 `--debug`를 사용할 수 있으며, 이 모드는 train과
validation을 각각 최대 256개 sample, 1 epoch로 제한한다.

## 5. 로그와 체크포인트

각 `--output-dir`에는 다음 파일이 생성된다.

```text
outputs/task2_baseline/
├── train.log
├── metrics.csv
├── best.pth
└── last.pth
```

- `train.log`: 학습 중 터미널에 표시되는 설정과 epoch 메시지를 기록한다.
- `metrics.csv`: 완료된 모든 epoch의 loss, learning rate, 원본/EMA Rank-1과
  mAP/mADM, 최고 epoch, 선택 metric, early-stopping counter와 소요 시간을 기록한다.
- `best.pth`: 선택한 validation metric이 개선될 때만 교체되는 추론·제출용 모델이다.
- `last.pth`: 매 epoch 교체되며 모델, EMA, optimizer, scheduler, AMP scaler,
  난수 상태와 early-stopping 상태를 포함한다.

체크포인트는 임시 파일에 먼저 저장한 뒤 원자적으로 교체한다. 같은 output
directory에서 새 학습을 시작하면 `train.log`와 `metrics.csv`는 새로 작성되고
체크포인트도 새 학습 결과로 교체된다. 실험을 보존하려면 실험마다 서로 다른
`--output-dir`을 사용한다.

## 6. 중단된 학습 재개

동일한 output directory의 `last.pth`에서 재개한다.

```bash
python examples/task2/sample_code_submission/run.py \
  --mode train_eval \
  --output-dir outputs/task2_baseline \
  --epochs 100 \
  --resume
```

`--epochs`는 추가 학습 횟수가 아니라 **전체 목표 epoch**다. 예를 들어
`last.pth`가 35 epoch에서 저장되었다면 위 명령은 36 epoch부터 시작한다.
명시적인 경로도 사용할 수 있다.

```bash
python examples/task2/sample_code_submission/run.py \
  --mode train_eval \
  --output-dir outputs/task2_baseline \
  --resume outputs/task2_baseline/last.pth
```

재개할 때는 `last.pth`와 짝을 이루는 `best.pth`가 같은 `--output-dir`에 있어야
하며 EMA 사용 여부도 기존 실행과 같아야 한다. 이전 버전의 `model_best.pth`나
현재의 `best.pth`에는 optimizer 상태가 없으므로 resume에는 사용할 수 없다.

## 7. 자주 쓰는 옵션

```bash
# early stopping을 20회 검증으로 변경
python examples/task2/sample_code_submission/run.py \
  --early-stopping-patience 20

# EMA 비활성화
python examples/task2/sample_code_submission/run.py --no-ema

# 검증 probability 캐시 사용
python examples/task2/sample_code_submission/run.py \
  --mode eval \
  --checkpoint outputs/task2_baseline/best.pth \
  --cache-val-probs

# 검증 메모리 사용량 감소
python examples/task2/sample_code_submission/run.py \
  --eval-batch-size 64 \
  --query-chunk-size 128
```

`--retrieval-interval N`으로 retrieval 검증 주기를 조절할 수 있다. Early
stopping patience는 epoch 수가 아니라 실제 retrieval 검증 횟수를 센다.

## 8. 제출 패키지 준비

Challenge ingestion은 `run.py`를 import하고 `rank_gallery()`를 호출한다. 학습한
최고 모델로 Codabench 업로드용 ZIP을 생성한다.

```bash
python examples/task2/sample_code_submission/package_submission.py \
  --checkpoint outputs/task2_baseline/best.pth
```

AttriVision checkpoint도 같은 명령으로 패키징할 수 있다.

```bash
python examples/task2/sample_code_submission/package_submission.py \
  --checkpoint outputs/attrivision_task2_hybrid/checkpoint_best.pth
```

A7 AttriVision checkpoint는 `--model attrivision_a7` preset으로 패키징한다. 이
옵션은 제출용 `run.py`에 A7과 동일한 `native52_category_nll`, temperature
`0.01`, `resize_pad_crop` 평가 전처리를 함께 저장한다. 기존 ConvNeXt 제출
경로와 일반 checkpoint 패키징 동작은 그대로 유지된다.

```bash
python examples/task2/sample_code_submission/package_submission.py \
  --checkpoint outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth \
  --model attrivision_a7 \
  --output submissions/attrivision_a7.zip
```

이 경우 packager가 OpenCLIP text encoder로 52개 고정 prompt embedding을 미리
계산하고, visual encoder weight만 제출용 checkpoint로 변환한다. 따라서 Codabench
환경에 `open_clip_torch`가 없어도 실행되며 원본 약 605 MB checkpoint는 약 352 MB로
줄어든다.

스크립트는 체크포인트 형식을 확인하고, ZIP 최상단에 `run.py`와
`metadata.yaml`이 오도록 구성하며, 가중치를 `assets/model_best.pth`라는 이름으로
포함한다. 생성 후 archive 구조와 SHA-256도 검증한다. ZIP 안에
`sample_code_submission/` 폴더가 한 겹 더 들어가면 Codabench가 `run.py`를 찾지
못하므로 폴더 자체가 아닌 **폴더의 내용**이 archive root에 있어야 한다.
`--output`을 생략하면 저장소의 `submissions/` 아래에 현재 지역 날짜와 시간을
사용한 `upar_task2_YYYYMMDD_HHMMSS.zip` 이름으로 생성된다. 특정 이름이
필요하면 `--output submissions/my_submission.zip`처럼 지정할 수 있다.

Codabench의 해당 competition에서 참가 등록과 약관 동의를 마친 뒤 **My
Submissions** 탭의 첨부 버튼으로 생성된 ZIP을 업로드한다. 제출 환경은 네트워크에
접근할 수 없으므로 checkpoint와 필요한 모든 모듈이 ZIP 안에 있어야 한다.

공식 실행 환경은 Python 3.11, PyTorch 2.4.1, CUDA 12.1 기반이며 전체 실행
제한은 1시간이다. 현재 phase 제한은 하루 3회, 사용자당 총 50회다. Leaderboard
대표 점수는 best가 아니라 **가장 최근의 valid submission**이므로 새 모델을
올릴 때 주의한다.

공식 ranking metric은 `mADM`이며 mAP, Rank-1/5/10과 mINP도 함께 계산된다.
로컬 학습 코드는 validation mAP를 기준으로 `best.pth`를 선택하므로 로컬 mAP가
좋아져도 공식 mADM이 반드시 같은 비율로 좋아지는 것은 아니다.

제출 API는 다음 함수로 구성된다.

- `load_model()`: checkpoint와 전처리 설정을 한 번 로드한다.
- `predict_attributes(gallery, attribute_names)`: ConvNeXt 제출에서 gallery별
  `[N, 40]` attribute probability를 반환한다.
- `rank_gallery(sample)`: ConvNeXt는 L1 distance를, AttriVision은 선택된
  cosine similarity 또는 Native52 distance 행렬을 반환한다. A7 preset으로
  패키징한 AttriVision은 `native52_category_nll` distance를 반환한다.

A7의 제출 ranking은 gallery의 52개 semantic-state cosine logit을 category별
softmax(`T=0.01`)로 바꾼 뒤, query의 12개 category target에 대한 평균
negative log-likelihood를 distance로 계산한다. 따라서 challenge ingestion이
표준 `run.py`의 `rank_gallery()`를 호출해도 A7 validation과 같은 rank 방향이
유지된다.

추론 환경은 `UPAR_DEVICE`, `UPAR_CHECKPOINT`, `UPAR_BATCH_SIZE`,
`UPAR_NUM_WORKERS`, `UPAR_AMP` 환경 변수로 조정할 수 있다. AttriVision 제출의
ranking/geometry를 점검할 때는 `UPAR_RETRIEVAL_SCORING`,
`UPAR_CATEGORY_TEMPERATURE`, `UPAR_AUGMENTATION`도 override할 수 있다.

## 9. 코드 구조

```text
sample_code_submission/
├── run.py                 # CLI 및 challenge용 공개 진입점
├── metadata.yaml          # challenge package metadata
├── assets/                # 제출용 checkpoint 위치
└── upar/
    ├── challenge.py       # ingestion API
    ├── attrivision_challenge.py # 의존성 없는 제출용 CLIP visual encoder
    ├── checkpoint.py      # best/last 저장, 로드 및 resume
    ├── cli.py             # CLI 옵션과 실행 모드
    ├── config.py          # 경로, 전처리 설정, device와 seed
    ├── data.py            # CSV 파싱, 이미지 경로, dataset과 transform
    ├── engine.py          # 학습, 검증, early stopping
    ├── modeling.py        # ConvNeXt-B, weighted BCE, EMA
    ├── retrieval.py       # probability, L1 distance, Rank-1/mAP
    └── tracking.py        # train.log와 metrics.csv
```

실험 산출물과 데이터셋, checkpoint는 용량이 크므로 Git에 커밋하지 않는다.
