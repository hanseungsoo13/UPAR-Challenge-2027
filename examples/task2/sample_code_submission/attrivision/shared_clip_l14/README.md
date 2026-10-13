# Shared CLIP ViT-L/14 E1

OpenAI CLIP Large로 E1-A → E1-B → E1-C/E1-D를 새로 학습하는 별도 실행 경로다.
현재 B/16 실험의 A7 FCE, category-complete 52-state prompts, 12-category
local FCE, Global/Local fusion(alpha=0.5), validation mADM 선택을 재사용한다.
설정은 `attrivision/configs/e1_shared_clip_l14.json`에 있다.

- OpenCLIP 모델: `ViT-L-14`, pretrained tag: `openai`.
- 모델 식별자: `openai/clip-vit-large-patch14`.
- 입력: 224×224. CLS를 제외한 256개 패치(16×16), visual hidden width 1024.
- 이미지와 텍스트 embedding: 768. Local head는 1024 → 768로 투영한다.
- 하나의 visual encoder forward에서 global embedding과 local patch tokens를 얻는다.
- OpenAI QuickGELU를 사전학습 초기화와 `pretrained=None` 체크포인트 복원에 모두 적용한다.
- 기본 batch: train 16 / eval 32, AMP 사용. `--batch-size`와
  `--eval-batch-size`로 조정할 수 있다. 배치 크기는 FCE의 배치 내 대조 대상도
  바꾸므로 B/16과 비교할 때 실제 사용한 값을 함께 기록한다.
- A는 기존 A7 기본값인 최대 100 epochs, B/C는 설정의 최대 30 epochs다.
  조기 종료가 적용된다. A의 설정은 `experiment_config.json`, B/C의 설정은
  `config.json`으로 각 출력 폴더에 저장된다.

B/16·B/32 체크포인트는 Large 단계 초기화에 사용할 수 없다. A는 OpenAI L/14
사전학습 가중치에서 시작하고, B는 A-Large, C는 B-Large 결과를 사용한다.
출력은 `outputs/BackboneUpgrade/E1_SharedCLIP_L14/`에 저장한다.

## 실행 위치

`environment.yml`이 있는 `UPAR-Challenge-2027` 저장소 루트에서 실행한다.
기존 프로젝트 환경과 `data/`의 학습·검증 데이터가 필요하다.

```bash
conda activate rws-upar-challenge
RUNNER=examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py
CONFIG=examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json
OUT=outputs/BackboneUpgrade/E1_SharedCLIP_L14
```

## E1-A — CLIP Large A7 전체 fine-tuning

처음 실행할 때 OpenAI 사전학습 가중치를 다운로드한다. 기존 A7과 같은
category-complete `paper_fce` 학습으로 이미지·텍스트 tower를 모두 갱신한다.
최적 체크포인트의 global-only 검증 결과를 `evaluation_val.json`에 저장한다.

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" a7 \
  --mode train_eval --data-root data --device cuda:0 \
  --batch-size 16 --eval-batch-size 32 \
  --output-dir "$OUT/E1-A"
```

## E1-B — CLIP 고정, local head 학습

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" train \
  --stage local --consistency-variant B0 \
  --init-checkpoint "$OUT/E1-A/checkpoint_best.pth" \
  --config "$CONFIG" --data-root data --device cuda:0 \
  --output-dir "$OUT/E1-B"
```

기본 명령은 전체 학습 데이터를 사용한다. 기존 B/16에서 holdout 실험을 했다면
같은 `--holdout-domain Market1501`(또는 `PA100k`, `PETA`)을 추가한다.
동일한 holdout과 seed로 B1(`--consistency-variant B1`) 또는
B2(`--consistency-variant B2`)를 실행할 수도 있다. 각 variant는
`--output-dir "$OUT/E1-B1"` / `"$OUT/E1-B2"`처럼 별도 출력 폴더를 쓴다.

## E1-C — shared vision과 local head 공동 fine-tuning

E1-B/B0 결과에서 시작해 텍스트 tower는 고정하고 visual tower와 global
temperature, local decoder를 갱신한다. 기본 learning rate는 vision 1e-5,
decoder 1e-4이며 loss는 global FCE + 0.25 × local FCE다.

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" train \
  --stage joint \
  --init-checkpoint "$OUT/E1-B/checkpoint_best.pth" \
  --config "$CONFIG" --data-root data --device cuda:0 \
  --output-dir "$OUT/E1-C"
```

선택적으로 `--vision-learning-rate 2e-6 --clip-feature-anchor-weight 0.1`을
추가하면 E1-B의 고정 global image feature를 teacher로 사용하는 anchor
loss가 적용된다. 이 경우 별도 출력 폴더를 지정한다. B/C는 global-only,
local-only, alpha=0.5 fusion을 모두 평가하고 fused mADM으로 최적 모델을 고른다.

## E1-D — CLIP과 cross-attention AttriQuery 전체 joint 학습

E1-B/B0 최적 체크포인트에서 시작한다. E1-C는 거치지 않아도 된다.
CLIP visual tower, text tower, global temperature와 전체 local head
(category queries, cross-attention, FFN, LayerNorm, output projection)를 모두
갱신한다. 52-state text embedding도 각 배치에서 gradient를 유지한 채 다시
계산해 local FCE가 text tower까지 전달되도록 한다.

Loss는 `1.0 × global A7 FCE + 0.25 × category-local FCE`다. Vision/text는
각각 1e-5, local decoder는 1e-4 learning rate를 사용한다. 첫 배치에서
global/local loss의 vision·text gradient 및 모든 local parameter의 gradient를
검사한다. Optimizer에는 학습 대상 파라미터가 정확히 한 번씩 포함되는지도
확인한다. `parameter_counts.json`에는 cross-attention, queries, FFN,
projection의 학습 파라미터 수를 기록한다.

다음 명령은 위 셸 변수 설정 없이 저장소 루트에서 그대로 실행할 수 있다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py e1-d \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data \
  --device cuda:0 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --vision-learning-rate 1e-5 \
  --text-learning-rate 1e-5 \
  --decoder-learning-rate 1e-4 \
  --global-loss-weight 1.0 \
  --local-loss-weight 0.25 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-D
```

`e1-d`는 E1-B 초기화 체크포인트를 필수로 받는다. 기존
`train --stage full --init-checkpoint ...`도 같은 E1-D 학습을 실행한다.
Global-only, local-only, fusion(alpha=0.5)을 평가하고 fused validation mADM으로
`checkpoint_best.pth`를 선택한다. 검증 결과는 `evaluation_val.json`에 저장한다.

E1-D 재개 시에는 위 명령에 다음 인자를 추가한다.

```bash
--resume outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-D/checkpoint_last.pth
```

## 중단된 학습 재개

같은 출력 폴더와 초기화 체크포인트, batch 및 학습 설정을 유지한다.

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" a7 \
  --mode train_eval --data-root data --device cuda:0 \
  --batch-size 16 --eval-batch-size 32 \
  --resume "$OUT/E1-A/checkpoint_last.pth" --output-dir "$OUT/E1-A"

CUDA_VISIBLE_DEVICES=0 python "$RUNNER" train \
  --stage local --consistency-variant B0 \
  --init-checkpoint "$OUT/E1-A/checkpoint_best.pth" \
  --resume "$OUT/E1-B/checkpoint_last.pth" \
  --config "$CONFIG" --data-root data --device cuda:0 --output-dir "$OUT/E1-B"

CUDA_VISIBLE_DEVICES=0 python "$RUNNER" train \
  --stage joint --init-checkpoint "$OUT/E1-B/checkpoint_best.pth" \
  --resume "$OUT/E1-C/checkpoint_last.pth" \
  --config "$CONFIG" --data-root data --device cuda:0 --output-dir "$OUT/E1-C"
```

## 재평가와 모델 점검

`--experiment`와 경로를 E1-A/B/C에 맞춰 바꾸면 각 최적 모델을 재평가한다.

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" eval \
  --experiment E1-C --checkpoint "$OUT/E1-C/checkpoint_best.pth" \
  --data-root data --device cuda:0 --eval-batch-size 32
```

모델 점검 명령은 무작위 L/14를 생성하고 공유 encoder, feature shape,
학습 파라미터, gradient, 체크포인트 저장·복원을 확인한다.
사전학습 다운로드와 데이터 로딩은 하지 않지만 실제 L/14 크기의 메모리가 필요하다.

```bash
CUDA_VISIBLE_DEVICES=0 python "$RUNNER" smoke --device cuda:0 --batch-size 1
```

이 경로는 학습과 평가용이다. 기존 제출 packager의 B/16·B/32 전용 visual
runtime에는 L/14 지원을 별도로 추가해야 한다.
