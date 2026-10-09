# AttriVision for UPAR Task 2

이 폴더는 WACV 2025 논문 **AttriVision: Advancing Generalization in
Pedestrian Attribute Recognition using CLIP**을 UPAR Challenge 2027 Task 2
(attribute-based person retrieval)에 맞게 구현한 독립 baseline이다. 기존
ConvNeXt baseline은 상위 폴더에 그대로 유지된다.

## 구현 범위

- OpenCLIP `ViT-B-32-quickgelu`와 OpenAI pretrained weight
- 224×224 image input, 77-token text input, 512차원 L2-normalized feature
- image encoder와 text encoder 전체 fine-tuning
- UPAR 40개 binary annotation을 12개 의미 범주의 52개 상태 prompt로 변환
- 남성, 긴 소매, 가방 없음처럼 원본에서 0으로 표현되는 complement 상태도 학습
- 52개 text prototype 전체에 대한 class-balanced sigmoid focal supervision
- 평가와 동일한 attribute-set 평균 descriptor의 양방향 contrastive supervision
- 논문 4.2절의 unique prompt/owner-pair FCE는 `paper_fce` ablation으로 유지
- `single`/`multi` semantic-state text sampling
- multi에서 선택된 모든 attribute phrase를 평균하지 않고 독립 text로 학습
- 양방향 CLIP loss와 Focal Cross-Entropy(FCE) loss
- diagonal target과 중복 attribute를 positive로 다루는 multi-positive target
- query의 12개 범주 상태 text feature를 평균한 ABPR descriptor
- LayerNorm/bias/temperature를 weight decay에서 제외한 AdamW
- 5 epoch linear warmup + 100 epoch cosine decay, 최소 50 epoch 학습
- 전체/vision/text trainable parameter와 epoch별 peak VRAM 기록
- UPAR 2027 공식 mADM/mAP/Rank-1/5/10/mINP와 동일 ranking의 semantic top-1
- `train.log`, `metrics.csv`, `checkpoint_best.pth`, `checkpoint_last.pth`, resume
- 기본 checkpoint 선택 기준은 UPAR 공식 `mADM`이며, epoch 로그의 `best_score`와
  `selection_metric`으로 선택 기준을 명시한다.
- `--paper-faithful` one-shot preset for the WACV recipe: 80 positive/negative
  prompts, multi-attribute FCE, QuickGELU, paper-like augmentation, paired-L1
  inference, and checkpoint selection by the official UPAR mADM

ConvNeXt, body-part pooling, SID, SetEncoder, learnable query, cross-attention,
별도 domain-generalization 모듈은 포함하지 않았다.

## 설치

저장소 루트에서 환경을 갱신한다. `environment.yml`에는
`open_clip_torch>=2.26,<3`이 추가되어 있다.

```bash
conda env update -n rws-upar-challenge -f environment.yml --prune
conda activate rws-upar-challenge
```

## 실행

### E0 FCE mask audit

A1의 train batch에서 실제로 sampling된 text candidate만 대상으로 FCE
positive mask의 false-negative/false-positive를 확인하려면 별도
e0_mask_audit/ 폴더의 audit를 실행한다. 이 진단은 GT, 기존 sampler,
기존 FCE mask 코드만 사용하며 checkpoint와 CLIP encoder를 로드하지 않는다.

~~~bash
python examples/task2/sample_code_submission/attrivision/e0_mask_audit/e0_mask_audit.py \
  --data-root data \
  --seeds 42 43 44 45 46
~~~

기본값은 A1의 category_complete + multi(max3) + multi_positive 설정이다.
결과는 outputs/attrivision_e0_mask_audit/summary.json과 CSV 파일에 저장된다.
owner-only diagonal을 비교하려면 --contrastive-target diagonal을 추가한다.
자세한 지표 정의와 파일 목록은 attrivision/e0_mask_audit/README.md를
참고한다.

모든 명령은 저장소 루트에서 실행한다. 물리 GPU 6번만 노출하면 프로그램
내부의 장치 번호는 `cuda:0`이다.

### 구현 sanity check

pretrained weight를 다운로드하지 않고 실제 ViT-B/32의 image/text forward와
Task 2 hybrid loss backward를 검사한다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --mode smoke \
  --device cuda:0
```

### 학습

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --task task2 \
  --model attrivision \
  --mode train \
  --device cuda:0 \
  --training-objective task2_hybrid \
  --prompt-mode category_complete \
  --batch-size 128 \
  --output-dir outputs/attrivision_task2_hybrid
```

첫 실행에서는 OpenAI CLIP pretrained weight를 다운로드하므로 인터넷 연결이
필요하다. 이후에는 로컬 cache를 사용한다.

### 평가

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --task task2 \
  --model attrivision \
  --mode eval \
  --device cuda:0 \
  --checkpoint outputs/attrivision_task2_hybrid/checkpoint_best.pth
```

기존 checkpoint를 재학습하지 않고, 각 속성의 positive/negative prompt로 40개
확률을 만든 뒤 공식 L1 distance로 평가하려면 다음처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --mode eval \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --retrieval-scoring paired_l1
```

속성 `a`에 대해 negative/positive prompt feature와 image feature의 cosine을 각각
`s_a^-`, `s_a^+`로 계산하고, checkpoint에 저장된 CLIP의 learned temperature를
사용해 `softmax([s_a^-, s_a^+] / T)[1]`을 `p_a(x)`로 만든다. 최종 ranking
distance는 `sum_a |q_a - p_a(x)|`이다. 고정 온도를 비교하고 싶으면 예를 들어
`--attribute-temperature 0.07`을 추가한다.

### 학습 후 평가

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --task task2 \
  --model attrivision \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/attrivision_task2_hybrid
```

### 논문식 재현 preset

논문에 맞춘 설정을 개별 옵션으로 다시 조합하지 않도록 preset을 제공한다.
이 preset은 40개 binary attribute 각각에 대해 presence/absence 문장을 만들고,
image마다 최대 3개 문장을 무작위로 선택하고, 기존 재현 run과 같은 batch size 32로
식 (2)~(6)의 diagonal FCE를 계산한다.
검증은 40개 paired prompt 확률의 공식 L1 retrieval로 하고, checkpoint는 mAP가
아닌 UPAR 공식 mADM이 가장 높은 epoch를 저장한다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --mode train_eval \
  --device cuda:0 \
  --paper-faithful \
  --batch-size 32 \
  --output-dir outputs/attrivision_paper_faithful
```

논문은 optimizer의 세부값, batch size, 전체 prompt 목록을 공개하지 않으므로
이 preset의 해당 값은 CLI에서 확인·변경할 수 있다. `--paper-faithful`이 강제로
고정하는 것은 논문에서 확인되는 구조적 요소와 평가 protocol이며, 데이터셋 버전이
UPAR 2024인지 현재 Challenge release인지에 따른 점수 차이는 별도로 남는다.

attribute sampling 수 `K=3`은 논문에 공개된 값이 아니라 기존 재현 실험을 위한
기본값이다. 논문의 미공개 K를 비교하려면 `--paper-multi-attributes K`를 사용한다.
예를 들어 `K=40`은 모든 binary state를 사용하지만, batch 안의 동일 prompt가
반복되어 diagonal loss의 false negative가 늘고 text encoder 메모리/시간도 크게
증가한다. 따라서 이 비교에서는 `--paper-contrastive-target multi_positive`와 작은
batch size를 함께 별도 실험하는 것이 안전하다.

```bash
--paper-multi-attributes 40 \
--paper-contrastive-target multi_positive \
--paper-batch-size 8
```

### Crop ablation

ConvNeXt E3의 강한 crop을 완화하는 효과를 Attrivision에서 확인하려면 A5를
실행한다. A5는 A3와 동일한 multi-sampling/diagonal target 설정을 유지하고,
학습 crop만 다음처럼 바꾼다.

```text
RandomResizedCrop(224, scale=(0.08, 1.0))
→ RandomResizedCrop(224, scale=(0.5, 1.0))
```

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run_attrivision_ablation.py \
  --experiment A5 \
  --mode train_eval \
  --device cuda:0
```

결과는 `outputs/attrivision_ablation/A5/`에 저장되며, A3와의 차이는
`outputs/attrivision_ablation/pairwise_differences.json`에서 확인할 수 있다.

A6는 가장 기본적인 A0를 기준으로 ConvNeXt E3의 순서를 직접 모사한다. A0와 A6는
training objective, sampling, target 등 모든 설정을 공유하고 crop 정책만 다르다.
Attrivision의 square 입력 제약에 맞춰 전체 이미지를 먼저 `224×224`로 resize한 뒤,
`Pad(10) → RandomCrop(224)`로 작은 translation만 허용한다. A6 validation은 전체
이미지를 `Resize((224,224))`만 적용한다. A6는 기존 Native52 soft-L1 validation과
공식 `mADM` 기준을 유지한다.

```text
Resize((224,224))
→ Pad(10)
→ RandomCrop((224,224))
```

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run_attrivision_ablation.py \
  --experiment A6 \
  --mode train_eval \
  --device cuda:0 \
  --output-root outputs/attrivision_ablation_mADM
```

이 실행의 결과는 `outputs/attrivision_ablation_mADM/A6/`에
저장된다. 기존 결과를 보존하려면 `--output-root`를 새 경로로 지정하면 된다.
기본 경로를 사용할 때 결과는 `outputs/attrivision_ablation/A0/`와
`outputs/attrivision_ablation/A6/`에
저장되며, 두 결과가 모두 있으면 `A6 - A0` 차이가
`outputs/attrivision_ablation/pairwise_differences.json`에 기록된다.

A6와 동일한 crop을 사용하되 Category-NLL ranking으로 평가하는 별도 A7은 다음과
같이 실행한다. A7은 `category_temperature=0.01`과 공식 `mADM` 기준으로 best
checkpoint를 선택한다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run_attrivision_ablation.py \
  --experiment A7 \
  --mode train_eval \
  --device cuda:0 \
  --output-root outputs/attrivision_ablation_mADM
```

### A7-mixed: mixed categorical/multi-label retraining

기존 A7은 52개 state 전체를 category-softmax로 처리한다. A7-mixed는 Task 2
annotation의 실제 cardinality에 맞춰 50개 state를 사용한다. Age/Gender/Glasses와
같은 single-label category에는 category CE를, Hair/Upper color/Lower color/Lower
type에는 class-balanced multi-label BCE를 적용한다. `age_unknown`, `hair_other`,
`lower_type_other`, `glasses_none`은 유지하고, Task 2 train/val에서 positive가
없는 두 색상 `unspecified` state는 제거한다. 학습과 평가는 동일한 mixed-state
query/gallery NLL을 사용한다.

기존 A7 checkpoint를 덮어쓰지 않도록 별도 output directory에 저장한다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run_a7_mixed.py \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/attrivision_ablation_mixed/A7-mixed
```

빠른 smoke test는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run_a7_mixed.py \
  --mode smoke \
  --device cuda:0 \
  --no-pretrained
```

결과물은 `checkpoint_best.pth`, `checkpoint_last.pth`, `metrics.csv`, `train.log`,
`validation_metrics.json`으로 `outputs/attrivision_ablation_mixed/A7-mixed/`에
생성된다. 주요 설정은 checkpoint metadata의 `prompt_mode=mixed_category`,
`training_objective=a7_mixed`, `validation_protocol=mixed_state_nll`에서 확인할 수
있다.

### 재개

```bash
CUDA_VISIBLE_DEVICES=6 python examples/task2/sample_code_submission/attrivision/run.py \
  --mode train_eval \
  --device cuda:0 \
  --output-dir outputs/attrivision_task2_hybrid \
  --resume
```

`--resume` 뒤에 경로를 생략하면 같은 output directory의
`checkpoint_last.pth`를 사용한다. `--epochs`는 추가 epoch가 아니라 전체 목표
epoch다.

## 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--epochs` | 100 | 최대 학습 epoch |
| `--clip-model` | `ViT-B-32-quickgelu` | OpenAI weight와 activation이 일치하는 CLIP |
| `--batch-size` | 128 | train image batch size |
| `--learning-rate` | `1e-5` | AdamW learning rate |
| `--augmentation` | `current` | `current`, `rrc_scale_050`, `resize_pad_crop`, `paper_like` crop policy |
| `--training-objective` | `task2_hybrid` | Task 2 hybrid 또는 논문식 `paper_fce` |
| `--prototype-loss-weight` | `0.25` | 전체 semantic prototype focal loss 가중치 |
| `--set-loss-weight` | `1.0` | attribute-set contrastive loss 가중치 |
| `--balance-max-weight` | `10.0` | 희소 state class-balance weight 상한 |
| `--loss` | `focal_clip` | `clip` 또는 `focal_clip` |
| `--focal-alpha` | `1.0` | FCE alpha |
| `--focal-gamma` | `2.0` | FCE gamma |
| `--text-sampling` | `single` | `single` 또는 `multi` |
| `--multi-attributes` | 3 | multi에서 image당 sampling할 최대 prompt 수 |
| `--prompt-mode` | `category_complete` | 12개 범주, positive-only, paper binary 상태 |
| `--contrastive-target` | `diagonal` | 논문식 owner pair 또는 `multi_positive` ablation |
| `--unique-prompts` | 꺼짐 | `paper_fce` 전용 unique prompt sampler |
| `--query-aggregation` | `mean` | query text feature aggregation |
| `--retrieval-scoring` | `cosine_set` | `cosine_set` 또는 40확률 공식 L1인 `paired_l1` |
| `--attribute-temperature` | learned CLIP T | `paired_l1` softmax의 고정 온도 override |
| `--warmup-epochs` | 5 | linear learning-rate warmup epoch |
| `--min-learning-rate` | `1e-7` | cosine decay의 최저 learning rate |
| `--minimum-training-epochs` | 50 | 이 epoch 전에는 early stopping 금지 |
| `--early-stopping-patience` | 20 | 선택 metric 미개선 evaluation 횟수 |

전체 옵션은 `python .../attrivision/run.py --help`로 확인한다. 논문에 명시되지
않은 optimizer, learning rate, batch size, alpha, gamma, sampling 개수 등은 모두
CLI에서 바꿀 수 있다.

## 학습 text와 loss

기본 `category_complete`는 40-bit row를 age, gender, hair, upper/lower length,
upper/lower color, lower type, backpack, bag, glasses, hat의 12개 범주로 해석한다.
예를 들어 `Gender-Female=0`은 단순히 버리는 대신 `a photo of a man`으로,
`Accessory-Bag=0`은 `a photo of a person without a bag`으로 변환한다. 색상처럼
annotation에 여러 값이 동시에 켜질 수 있는 범주는 그 값을 모두 보존한다.
`binary_positive`는 이전처럼 1인 40개 속성만 사용하는 ablation이다.

기본 `task2_hybrid`는 52개 prompt를 batch마다 한 번씩 text encoder에 통과시킨다.
첫 번째 항은 모든 image×state pair에 class-balanced sigmoid focal loss를 적용하므로
owner 외의 실제 positive를 negative로 취급하지 않는다. 두 번째 항은 각 이미지의
전체 true state embedding을 평균·정규화해 query descriptor를 만들고 image feature와
양방향 contrastive loss를 계산한다. 이 descriptor는 평가와 제출에서 사용하는 것과
동일하다. 따라서 학습과 Task 2 ranking objective가 직접 정렬된다.

`paper_fce`에서는 `single`이 image의 참인 semantic state 중 하나를 무작위로 골라 하나의 짧은
prompt를 만든다. `multi`는 여러 state를 고르고 각 문장을 독립적으로
text encoder에 통과시킨다. 학습 중에는 이 feature들을 평균하지 않으며 여러 상태를
한 문장으로 이어 붙이지도 않는다. 따라서 image batch가 `B`, image당 선택 수가
`K`이면 image feature는 `[B,512]`, text feature는 `[T,512]`, similarity는
`[B,T]`이고 일반적으로 `T=B×K`다.

`paper_fce + diagonal`은 각 sampled text를 그 text의 원래 image와 짝지어 논문 식
(2)~(6)의 양방향 FCE를 계산한다. `--unique-prompts` sampler는 희소한 state를
우선 배정하면서 batch 안의 phrase 중복을 막는다. 이 조건 때문에 실제 batch가
요청 크기보다 작아질 수 있으며, 실제 평균 batch와 CUDA peak allocated/reserved
메모리는 매 epoch `train.log`와 `metrics.csv`에 기록된다.

단, `--paper-faithful`의 `paper_binary`에서는 한 이미지 안에서 같은 state를 두 번
뽑지 않는 `random.sample`을 사용한다. 서로 다른 이미지가 같은 `no ...` 문장을
공유하는 것은 서로 다른 `(image, label)` pair이므로 허용한다. 80개 state 전체를
batch 전역에서 강제로 중복 금지하면 기본 batch에서 불필요하게 batch가 쪼개진다.

`paper_fce + multi_positive`에서는 image `i`의 positive label과 text sample `j`가 선택한
semantic state가 하나라도 겹치면 `positive_mask[i,j]=1`이다. 방향별 loss는 각 anchor의
positive log-probability 평균을 사용한다. `diagonal`은 각 text를 그 text가
sampling된 원래 image에만 연결한다. `single`에서는 vanilla CLIP의 정사각
diagonal과 같다. 흔한 complement state가 batch 대부분을 positive로 만드는 문제 때문에
`multi_positive`는 더 이상 기본값이 아니다.

## 평가 방식

기본 모드는 각 query의 40-bit 값을 12개 범주 상태로 완성하고 해당 prompt의 512-D
feature를 평균한 뒤 normalize한다. 따라서 query에서 0인 값도 `man`, `without a bag`
같은 검색 조건으로 사용된다. gallery image 역시 512-D로 normalize한 뒤 cosine
similarity를 계산하며, 로컬 evaluator에는
`distance = -similarity`를 전달한다. annotation과 Rank-1/mAP 계산은 기존 `upar`
모듈을 재사용한다.

`--retrieval-scoring paired_l1`은 이 평균 descriptor를 사용하지 않는다. 40개
attribute 각각에 positive/negative 문장을 하나씩 배정해 총 80개 text feature를
한 번 계산하고, 각 gallery image를 40개의 2-class softmax 확률로 바꾼다. 이후
UPAR query의 원래 40-bit 순서로 열을 맞춘 뒤 공식 L1 distance를 계산한다.

같은 방식을 Codabench 제출 ZIP에 넣으려면 packaging할 때 scoring 방식을 명시한다.

```bash
python examples/task2/sample_code_submission/package_submission.py \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --retrieval-scoring paired_l1
```

이 ZIP에는 fine-tuned text encoder가 만든 paired prompt feature와 learned temperature가
함께 저장되므로 Codabench의 network-disabled 환경에서도 동일하게 추론한다. `paper_binary`
checkpoint는 반드시 `--retrieval-scoring paired_l1`로 패키징한다.

A7 checkpoint를 표준 제출 `run.py`로 사용할 때는 다음 preset을 사용한다. 이 preset은
Category-NLL ranking(`T=0.01`)과 A7의 `resize_pad_crop` 평가 geometry를 ZIP 안에
저장하므로, 제출 adapter에서도 학습 중 best checkpoint를 고른 동일한 rank 방식을
사용한다.

```bash
python examples/task2/sample_code_submission/package_submission.py \
  --checkpoint outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth \
  --model attrivision_a7 \
  --output submissions/attrivision_a7.zip
```

## Paired-L1 진단 실험

새로 학습하지 않고 GT oracle, 속성별 예측 품질, 40×40 cross-AUROC mapping,
nearest semantic-query recovery를 한 번에 검사한다.

```bash
CUDA_VISIBLE_DEVICES=9 python \
  examples/task2/sample_code_submission/attrivision/diagnose.py \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --eval-batch-size 512 \
  --num-workers 8
```

기본 산출물 경로는 checkpoint 옆의 `diagnostics/`다. `summary.json`, 실제 retrieval에
사용한 `pred_probs.npy`, 속성별 CSV, AUROC matrix CSV/NPY를 저장한다. gallery를 다시
encoding하지 않고 CPU 진단만 재실행하려면 저장된 probability를 넘긴다.

```bash
CUDA_VISIBLE_DEVICES=9 python \
  examples/task2/sample_code_submission/attrivision/diagnose.py \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --predictions outputs/attrivision_category/diagnostics/pred_probs.npy
```

## Inference formulation 4종 비교

checkpoint를 재학습하지 않고 동일한 validation gallery feature로 positive text
aggregation(A), 독립 positive/negative prompt(B), 학습에 사용한 exact 52-state의
category softmax(C), 같은 52-state raw score projection(D)을 비교한다.

```bash
CUDA_VISIBLE_DEVICES=9 python \
  examples/task2/sample_code_submission/attrivision/compare_inference.py \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --eval-batch-size 512 \
  --num-workers 8
```

D는 기본적으로 validation 전체에서 attribute 차원별 min-max normalization을 적용한다.
`--raw-normalization sigmoid` 또는 `--raw-normalization none`으로 바꿀 수 있다.
첫 실행은 `inference_comparison/gallery_features.npy`를 저장한다. 이후에는 아래처럼
이를 명시해 image encoding 없이 네 formulation만 다시 비교할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=9 python \
  examples/task2/sample_code_submission/attrivision/compare_inference.py \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --gallery-features outputs/attrivision_category/inference_comparison/gallery_features.npy
```

결과 폴더에는 `summary.json`, `summary.csv`, 방식별 40-D prediction NPY와 속성별
AUROC/AP/F1 CSV가 저장된다. A의 retrieval은 기존 normalized positive-text descriptor를
그대로 사용하며, A에는 확률 정의가 없으므로 40-D 진단 지표에만 validation min-max
score를 사용한다.

### Native category temperature sweep

위 비교에서 저장한 gallery cache를 재사용해 C 방식의 temperature만 바꿀 수도 있다.
이 명령은 image encoder를 호출하지 않으며, 52개 text feature는 처음 한 번만 인코딩해
`native_state_features.npy`로 저장한다.

```bash
CUDA_VISIBLE_DEVICES=9 python \
  examples/task2/sample_code_submission/attrivision/temperature_sweep.py \
  --device cuda:0 \
  --checkpoint outputs/attrivision_category/checkpoint_best.pth \
  --gallery-features outputs/attrivision_category/inference_comparison/gallery_features.npy
```

기본 sweep는 `0.01 0.02 0.05 0.1 0.2 0.5 1.0`이며, `temperature_sweep/` 아래에
`temperature_sweep.csv`와 JSON 결과를 남긴다. 이미 text cache가 있으면
`--native-text-features PATH`로 지정할 수 있다.

## 논문 재현상의 불확실성

논문은 여러 개의 attribute phrase를 image마다 사용한다고 설명하지만 식 (1)~(6)은
batch diagonal pair 형태와 image 내부의 여러 label 형태를 혼용하고, optimizer,
batch size, epoch, alpha/gamma, 실제 phrase 전체 목록을 공개하지 않는다. 따라서 이
구현은 식 (2)~(6)과 4.2절의 해석을 `--training-objective paper_fce`로 보존한다.
하지만 UPAR label에서 선택된 prompt는 평균 9.43개 batch image에 실제로 참이었고,
owner-only diagonal은 sampled pair의 81.7%에서 false negative를 만들었다. 이 경로는
loss가 감소하면서 retrieval이 붕괴했으므로 기본값으로 사용하지 않는다. 기본
`task2_hybrid`는 논문의 개별 짧은 phrase와 전체 CLIP fine-tuning을 유지하면서 이
ambiguity를 제거한 Task 2 확장이다.

기존 `ViT-B-32` 실행은 OpenAI pretrained weight가 기대하는 QuickGELU 대신 GELU를
사용했다. 현재 기본 architecture, optimizer parameter group, objective가 모두 달라졌으므로
기존 `checkpoint_last.pth`에서는 resume하지 말고 새 output directory에서 처음부터
학습한다. 이전 `checkpoint_best.pth`의 평가와 제출 패키징은 계속 지원한다.

또한 논문 4.4절은 “original CLIP preprocessing”과 “ImageNet statistics”를 함께
서술한다. 여기서는 요청 사양과 실제 OpenAI CLIP 관례에 맞춰 CLIP mean/std를
사용한다. 논문은 PAR의 image→attribute retrieval을 주로 설명하지만 이 코드는 Task
2의 attribute-set→gallery retrieval을 위해 범주별 state feature 평균을 query로 쓰는
명시적 확장이다. 원 요청의 active-only 표현은 `--prompt-mode binary_positive`로
계속 비교할 수 있다.

## 파일 구조

```text
attrivision/
├── run.py                         # 실행 진입점
├── cli.py                         # CLI와 sanity check
├── diagnose.py                    # Task2 mapping/prediction 진단
├── compare_inference.py           # A/B/C/D inference formulation 비교
├── temperature_sweep.py           # C 방식 temperature sweep (image cache 재사용)
├── checkpoint.py                  # best/last 저장과 resume
├── tracking.py                    # train.log와 metrics.csv
├── transforms.py                  # CLIP train/eval preprocessing
├── configs/attrivision.yaml       # 기본 설정 참고본
├── models/attrivision.py          # OpenCLIP ViT-B/32 wrapper
├── datasets/
│   ├── attribute_prompts.py       # 40-bit→12개 범주/52개 state prompt
│   └── upar_abpr.py               # dataset과 text sampling collator
├── losses/
│   ├── focal_clip_loss.py         # 논문식 CLIP/FCE ablation
│   └── task2_hybrid_loss.py       # prototype focal + set contrastive loss
└── engine/
    ├── trainer_attrivision.py     # 학습, early stopping, logging
    └── evaluator_abpr.py          # query/gallery encoding과 Task2 평가
```
