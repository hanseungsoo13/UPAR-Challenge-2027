# AttriVision A7 전용 학습·평가·제출

이 디렉터리는 UPAR Challenge 2027 Task 2에서 검증된 **AttriVision A7**만
실행하기 위한 진입점이다. 여러 AttriVision ablation 중 설정을 고를 필요 없이
[`run.py`](run.py) 하나로 smoke test, 학습, 평가, 재개, Codabench 제출 ZIP 생성을
처리한다.

## 기준 성능

전체 validation split(33,407 images, 3,462 queries)에서 선택된 epoch 8
checkpoint의 결과다.

| Metric | Value |
|---|---:|
| mADM | 0.68627 |
| mAP | 0.30377 |
| Rank-1 | 0.32611 |
| Rank-5 | 0.52744 |
| Rank-10 | 0.61843 |

평가는 Native52 Category-NLL, temperature `0.01`을 사용한다.

## 잠긴 A7 설정

`run.py`는 아래 설정을 항상 적용한다. 다른 AttriVision 실험으로 우연히 바뀌지
않도록 해당 항목은 CLI 옵션으로 노출하지 않는다.

- OpenAI CLIP `ViT-B-32-quickgelu`, input 224×224
- 12 categories / 52 semantic states (`category_complete`)
- sampled single-state FocalCLIP loss
- `multi_positive` contrastive target
- `Resize(224×224) → Pad(10) → RandomCrop(224×224)` 학습 augmentation
- Native52 Category-NLL validation, temperature `0.01`
- official `mADM` best-checkpoint selection

## 1. 환경과 데이터

저장소 루트에서 실행한다.

```bash
conda env create -f environment.yml
conda activate rws-upar-challenge
python download_datasets.py
```

이미 환경이 있다면 다음처럼 갱신할 수 있다.

```bash
conda env update -n rws-upar-challenge -f environment.yml --prune
```

기본 데이터 경로는 `data/`다. 다른 위치를 쓰면 모든 명령에
`--data-root /path/to/data`를 추가한다.

## 2. Smoke test

pretrained weight를 다운로드하지 않고 A7 모델, augmentation, loss, backward
경로를 확인한다.

```bash
python examples/task2/attrivision_a7/run.py \
  --mode smoke \
  --device cpu \
  --no-pretrained
```

GPU 6번을 사용할 때 프로세스 내부 장치는 `cuda:0`이다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/attrivision_a7/run.py \
  --mode smoke \
  --device cuda:0 \
  --no-pretrained
```

## 3. 학습 후 평가

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/attrivision_a7/run.py \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/attrivision_a7
```

기본 학습은 최대 100 epochs이며 validation mADM으로 best checkpoint를 고른다.
결과 디렉터리에는 다음 파일이 생성된다.

```text
outputs/attrivision_a7/
├── config.json
├── train.log
├── metrics.csv
├── checkpoint_best.pth
├── checkpoint_last.pth
└── validation_metrics.json
```

학습만 실행하려면 `--mode train`을 사용한다.

## 4. 중단된 학습 재개

같은 output directory의 `checkpoint_last.pth`에서 이어서 실행한다.
`--epochs`는 추가 epoch 수가 아니라 최종 목표 epoch다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/attrivision_a7/run.py \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/attrivision_a7 \
  --epochs 100 \
  --resume
```

다른 위치의 checkpoint는 `--resume /path/to/checkpoint_last.pth`로 지정한다.

## 5. 기존 checkpoint 평가

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/attrivision_a7/run.py \
  --mode eval \
  --device cuda:0 \
  --checkpoint outputs/attrivision_a7/checkpoint_best.pth \
  --output-dir outputs/attrivision_a7
```

평가 결과는 `validation_metrics.json`에 저장된다.

## 6. Codabench 제출 ZIP 생성

```bash
python examples/task2/attrivision_a7/run.py \
  --mode package \
  --checkpoint outputs/attrivision_a7/checkpoint_best.pth \
  --submission-output submissions/attrivision_a7.zip
```

패키저는 research checkpoint에서 visual encoder와 고정된 Native52 text feature를
추출한다. 생성된 ZIP은 다음 구조를 가지며 `run.py`가 ZIP 최상단에 놓인다.

```text
attrivision_a7.zip
├── run.py
├── metadata.yaml
├── assets/model_best.pth
└── upar/...
```

제출용 `run.py`는 Codabench ingestion이 import하는
`load_model`, `predict_attributes`, `rank_gallery` API를 제공한다. OpenCLIP text
encoder는 제출 환경에서 필요하지 않다. 패키징이 끝나면 archive 구조와 checkpoint
형식을 다시 읽어 검증하고 ZIP의 SHA-256을 출력한다.

> `checkpoint_best.pth`, 데이터셋, `outputs/`, `submissions/`는 용량과 라이선스
> 때문에 Git에 포함되지 않는다. 팀원에게 기존 epoch 8 checkpoint를 공유하려면
> 별도 스토리지에 올린 뒤 위 명령의 `--checkpoint` 경로로 지정해야 한다.

## 7. 주요 옵션

```bash
python examples/task2/attrivision_a7/run.py --help
```

GPU 메모리가 부족하면 `--batch-size`와 `--eval-batch-size`를 줄인다. 빠른 데이터
경로 확인에는 `--debug`를 사용할 수 있지만 이 결과는 정식 성능 비교에 사용하지
않는다. `--no-pretrained`는 smoke test에서만 허용되며 실제 학습은 검증된 OpenAI
pretrained initialization을 항상 사용한다.
