# Shared CLIP ViT-L/14 E1

OpenAI CLIP Large로 E1-A → E1-B → E1-C와 독립적인 E1-D를 새로 학습하는 별도 실행 경로다.
현재 B/16 실험의 A7 FCE, category-complete 52-state prompts, 12-category
local FCE, Global/Local fusion(alpha=0.5), validation mADM 선택을 재사용한다.
설정은 `attrivision/configs/e1_shared_clip_l14.json`에 있다.

- OpenCLIP 모델: `ViT-L-14`, pretrained tag: `openai`.
- 모델 식별자: `openai/clip-vit-large-patch14`.
- 입력: 224×224. CLS를 제외한 256개 패치(16×16), visual hidden width 1024.
- 이미지와 텍스트 embedding: 768. Local head는 1024 → 768로 투영한다.
- 하나의 visual encoder forward에서 global embedding과 local patch tokens를 얻는다.
- OpenAI QuickGELU를 사전학습 초기화와 `pretrained=None` 체크포인트 복원에 모두 적용한다.
- 기본 microbatch: train 16 / eval 32, AMP 사용. Gradient Cache는 16개씩
  8개 microbatch의 전체 손실을 계산해 effective batch 128로 한 번 갱신한다.
  `--batch-size`와 `--eval-batch-size`로 조정할 수 있다.
- A는 기존 A7 기본값인 최대 100 epochs, B/C는 설정의 최대 30 epochs다.
  조기 종료가 적용된다. A의 설정은 `experiment_config.json`, B/C의 설정은
  `config.json`으로 각 출력 폴더에 저장된다.

B/16·B/32 체크포인트는 Large 단계 초기화에 사용할 수 없다. A는 OpenAI L/14
사전학습 가중치에서 시작하고, B는 A-Large, C는 B-Large 결과를 사용한다.
출력은 `outputs/BackboneUpgrade/E1_SharedCLIP_L14/`에 저장한다.

## Gradient Cache와 effective batch

E1-A/B/C/D 모두 Gradient Cache를 지원한다. 기본값은 `--batch-size 16
--grad-cache-steps 8`로 effective batch 128이다. 각 microbatch의 CLIP/AttriQuery
forward를 차례로 실행해 feature만 보관하고, 128개 전체의 FCE 손실을 한 번 계산한
뒤 encoder를 다시 실행해 gradient를 전달한다. 따라서 GPU activation 메모리는
microbatch 16 기준으로 유지하면서 전체 128개가 서로 대조 대상이 된다.

메모리가 더 부족하면 두 값을 함께 바꿔 effective batch 128을 유지한다.

| `--batch-size` | `--grad-cache-steps` | effective batch |
|---:|---:|---:|
| 16 | 8 | 128 |
| 8 | 16 | 128 |
| 4 | 32 | 128 |

각 묶음마다 optimizer와 scheduler를 한 번 갱신한다. 마지막 묶음은 데이터 수에
따라 128보다 작을 수 있다. encoder를 다시 계산하므로 학습 시간은 늘고, 전체
effective batch의 feature와 FCE logits를 담는 메모리는 남는다. 끄려면
`--grad-cache-steps 1`을 지정한다.
검증 batch는 별도이므로 검증 중 메모리가 부족하면 `--eval-batch-size 16` 또는
`8`로 낮춘다.

## 실행 위치

`environment.yml`이 있는 `UPAR-Challenge-2027` 저장소 루트에서 실행한다.
기존 프로젝트 환경과 `data/`의 학습·검증 데이터가 필요하다.

```bash
conda activate rws-upar-challenge
```

## E1-A — CLIP Large A7 전체 fine-tuning

처음 실행할 때 OpenAI 사전학습 가중치를 다운로드한다. 기존 A7과 같은
category-complete `paper_fce` 학습으로 이미지·텍스트 tower를 모두 갱신한다.
최적 체크포인트의 global-only 검증 결과를 `evaluation_val.json`에 저장한다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py a7 \
  --mode train_eval --data-root data --device cuda:0 \
  --batch-size 16 --grad-cache-steps 8 --eval-batch-size 32 \
  --distill-weight 0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-A-GC128
```

## E1-B — CLIP 고정, local head 학습

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py train \
  --stage local --consistency-variant B0 \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-A-GC128/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data --device cuda:0 --batch-size 16 --grad-cache-steps 8 --eval-batch-size 32 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B-GC128
```

기본 명령은 전체 학습 데이터를 사용한다. 기존 B/16에서 holdout 실험을 했다면
같은 `--holdout-domain Market1501`(또는 `PA100k`, `PETA`)을 추가한다.
동일한 holdout과 seed로 B1(`--consistency-variant B1`) 또는
B2(`--consistency-variant B2`)를 실행할 수도 있다. 각 variant는
`outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B1-GC128` /
`outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B2-GC128`처럼 별도 출력 폴더를 쓴다.

## E1-C — shared vision과 local head 공동 fine-tuning

E1-B/B0 결과에서 시작해 텍스트 tower는 고정하고 visual tower와 global
temperature, local decoder를 갱신한다. 기본 learning rate는 vision 1e-5,
decoder 1e-4이며 loss는 global FCE + 0.25 × local FCE다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py train \
  --stage joint \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B-GC128/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data --device cuda:0 --batch-size 16 --grad-cache-steps 8 --eval-batch-size 32 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-C-GC128
```

선택적으로 `--vision-learning-rate 2e-6 --clip-feature-anchor-weight 0.1`을
추가하면 E1-B의 고정 global image feature를 teacher로 사용하는 anchor
loss가 적용된다. 이 경우 별도 출력 폴더를 지정한다. B/C는 global-only,
local-only, alpha=0.5 fusion을 모두 평가하고 fused mADM으로 최적 모델을 고른다.

## E1-D — CLIP과 cross-attention AttriQuery 전체 joint 학습

E1-A처럼 OpenAI pretrained CLIP L/14에서 바로 시작한다. AttriQuery local
head는 새로 무작위 초기화하며 E1-A/B/C 결과 없이 독립적으로 학습할 수 있다.
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

다음 명령도 저장소 루트에서 그대로 실행한다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py e1-d \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data \
  --device cuda:0 \
  --batch-size 16 --grad-cache-steps 8 \
  --eval-batch-size 32 \
  --vision-learning-rate 1e-5 \
  --text-learning-rate 1e-5 \
  --decoder-learning-rate 1e-4 \
  --global-loss-weight 1.0 \
  --local-loss-weight 0.25 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-D-GC128
```

`e1-d`와 `train --stage full`은 모두 pretrained CLIP에서 E1-D를 시작한다.
`--init-checkpoint`는 허용하지 않는다. 첫 실행 시 OpenAI 사전학습 가중치를
다운로드하며 저장 설정과 체크포인트에 `init_source=pretrained`를 기록한다.
Global-only, local-only, fusion(alpha=0.5)을 평가하고 fused validation mADM으로
`checkpoint_best.pth`를 선택한다. 검증 결과는 `evaluation_val.json`에 저장한다.

E1-D 재개 시에는 위 명령에 다음 인자를 추가한다. 이전 E1-B 초기화 방식으로
학습한 폴더가 있다면 새 출력 폴더에서 pretrained E1-D를 시작해야 한다.

```bash
--resume outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-D-GC128/checkpoint_last.pth
```

## 중단된 학습 재개

같은 출력 폴더와 초기화 체크포인트, batch 및 학습 설정을 유지한다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py a7 \
  --mode train_eval --data-root data --device cuda:0 \
  --batch-size 16 --grad-cache-steps 8 --eval-batch-size 32 \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-A-GC128/checkpoint_last.pth \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-A-GC128

CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py train \
  --stage local --consistency-variant B0 \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-A-GC128/checkpoint_best.pth \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B-GC128/checkpoint_last.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data --device cuda:0 --batch-size 16 --grad-cache-steps 8 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B-GC128

CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py train \
  --stage joint --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-B-GC128/checkpoint_best.pth \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-C-GC128/checkpoint_last.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_l14.json \
  --data-root data --device cuda:0 --batch-size 16 --grad-cache-steps 8 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-C-GC128
```

## 재평가와 모델 점검

`--experiment`와 경로를 E1-A/B/C에 맞춰 바꾸면 각 최적 모델을 재평가한다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py eval \
  --experiment E1-C --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_L14/E1-C-GC128/checkpoint_best.pth \
  --data-root data --device cuda:0 --eval-batch-size 32
```

모델 점검 명령은 무작위 L/14를 생성하고 공유 encoder, feature shape,
학습 파라미터, gradient, 체크포인트 저장·복원을 확인한다.
사전학습 다운로드와 데이터 로딩은 하지 않지만 실제 L/14 크기의 메모리가 필요하다.

```bash
conda activate rws-upar-challenge && CUDA_VISIBLE_DEVICES=1 python examples/task2/sample_code_submission/attrivision/shared_clip_l14/run.py smoke --device cuda:0 --batch-size 1
```

이 경로는 학습과 평가용이다. 기존 제출 packager의 B/16·B/32 전용 visual
runtime에는 L/14 지원을 별도로 추가해야 한다.
