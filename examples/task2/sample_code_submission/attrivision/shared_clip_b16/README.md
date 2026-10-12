# Shared CLIP ViT-B/16 E1

This isolated experiment replaces the B/32 image and text towers used by the
new E1 runs with one OpenAI CLIP ViT-B/16 model. Existing A7/AttriQuery source,
checkpoints, and E2/E3 implementations are left in place.

## Architecture and reused implementation

- Pretraining source: `openai/clip-vit-base-patch16`, loaded through the
  repository's existing OpenCLIP `AttriVision` wrapper as
  `ViT-B-16-quickgelu` to match the OpenAI weight activation.
- The same `base.clip.visual` instance produces the normalized global CLIP
  feature and the spatial patch sequence in one visual forward pass.
- At 224×224, the visual transformer provides 196 spatial tokens of width 768
  after removing CLS. The global image and text features each have width 512.
- Local category queries attend to those 196 patch tokens through the existing
  AttriQuery `CategoryCrossAttention`. Its output projection maps 768-D visual
  features to the 512-D CLIP text space; all 52 state prototypes come from the
  same B/16 text encoder.
- A7 full fine-tuning reuses the existing `AttriVision` trainer, A7
  `paper_fce` objective, category-complete 52-state prompts, resize/pad/crop
  augmentation, warmup/cosine AdamW schedule, Native52 Category-NLL evaluator,
  and validation mADM checkpoint selection.
- Local training reuses `CategoryLocalFCELoss`, the 12-category decoder, and
  the established Global/Local log-probability fusion evaluator. The default
  fusion alpha stays fixed at 0.5.

The isolated configuration is
[`e1_shared_clip_b16.json`](../configs/e1_shared_clip_b16.json). E1-0 and E1-B/B0
share the same local objective, epoch count, optimizer, decoder, and evaluation
protocol. Their only intended difference is whether the frozen B/16 encoders
start from pretrained CLIP or E1-A. E1-C starts from the E1-B/B0 baseline, keeps the E1-A text
tower frozen, and tunes the shared visual tower at 1e-5 plus the local
projector/decoder at 1e-4. Its objective is `1.0 × A7 paper_fce + 0.25 × local
FCE`. E1-D also starts from E1-B, then jointly fine-tunes every CLIP B/16
parameter (vision tower, text tower, and global temperature) plus the complete
local decoder. It uses separate 1e-5 vision/text and 1e-4 decoder learning
rates with the same global/local objective. Both E1-D losses are checked on the
first training batch to confirm gradients reach both CLIP towers. E1-E uses
the same full-joint training but starts directly from OpenAI pretrained B/16
with a new local decoder, so it does not depend on E1-A or E1-B.

Training writes an independent configuration sidecar (`experiment_config.json`
for E1-A and `config.json` for E1-0/B/C/D/E), parameter counts, epoch metrics,
`checkpoint_best.pth`, resumable `checkpoint_last.pth`, and
`evaluation_val.json` files under `outputs/BackboneUpgrade/E1_SharedCLIP_B16/`.
All shared-model checkpoints record their model dimensions, initialization, and
single-encoder invariant. Loading a B/32 checkpoint as B/16 is rejected.

## Commands

Run from the repository root after activating the project environment. These
commands are provided for the user's later execution; no Python command,
download, training, validation, or evaluation was run while implementing E1.

### Shared encoder smoke checks

The smoke command uses randomly initialized B/16 weights and does not download
pretrained weights or read dataset files.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py smoke --device cpu
```

### E1-A — A7 full fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py a7 \
  --mode train_eval \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A
```

This initializes both B/16 towers from OpenAI pretrained weights and selects
the best checkpoint by validation mADM. The run writes
`outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/checkpoint_best.pth` and its
global-only `evaluation_val.json`.

Re-evaluate that best checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-A \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

To resume E1-A from its latest checkpoint, keep the same output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py a7 \
  --mode train \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/checkpoint_last.pth \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A
```

### E1-B — AttriQuery consistency study

All three runs load the same E1-A checkpoint, freeze both CLIP towers, start
the local queries from the same seed (`42` by default), and keep the FCE,
optimizer, batch size, epoch count, and validation protocol fixed. The example
holds out `Market1501` from training; replace it with `PA100k` or `PETA` to
screen another unseen source. The full official validation set still selects
the best checkpoint by fused mADM. Domain metrics are reported separately using
each source's own validation gallery.

| Variant | Training objective | Output directory |
| --- | --- | --- |
| B0 | Local FCE | `E1-B` |
| B1 | Local FCE + 0.1 × feature consistency | `E1-B1` |
| B2 | Local FCE + 0.1 × feature consistency + 0.1 × attention consistency | `E1-B2` |

The deterministic paired style view changes exposure and contrast only. It
does not move pixels, so attention maps use identical 14×14 coordinates. The
evaluation JSON records official validation and held-out domain mADM, Rank-1,
and mAP, plus mean per-category query-feature and attention cosine similarity.

```bash
E1A=outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/checkpoint_best.pth
CONFIG=examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json

CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage local --consistency-variant B0 --holdout-domain Market1501 \
  --init-checkpoint "$E1A" --config "$CONFIG" --data-root data --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B

CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage local --consistency-variant B1 --holdout-domain Market1501 \
  --init-checkpoint "$E1A" --config "$CONFIG" --data-root data --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B1

CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage local --consistency-variant B2 --holdout-domain Market1501 \
  --init-checkpoint "$E1A" --config "$CONFIG" --data-root data --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B2
```

Consistency strengths can be overridden with `--feature-consistency-weight`
and `--attention-consistency-weight`. B0 requires both to be zero; B1 requires
a positive feature weight and zero attention weight; B2 requires both weights
to be positive. Each run writes `evaluation_val.json`, `metrics.csv`, and
`checkpoint_best.pth` in its output directory. B0 keeps the established E1-B
checkpoint name so E1-C and E1-D can continue to initialize from that baseline.

To resume a run, pass its original `--consistency-variant`, `--holdout-domain`,
initial E1-A checkpoint, and output directory together with its
`checkpoint_last.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage local --consistency-variant B0 --holdout-domain Market1501 \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_last.pth \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B
```

### E1-C — shared vision and local decoder joint fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage joint \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C
```

Both the global A7 FCE loss and local category FCE loss are checked on the first
training batch to confirm that they reach the same trainable B/16 visual
parameters. To resume, retain the original E1-B initialization path and use
`--resume .../E1-C/checkpoint_last.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage joint \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_best.pth \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C/checkpoint_last.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C
```

### E1-D — full CLIP B/16 and local decoder joint fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage full \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_best.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D
```

This updates all parameters in both B/16 towers and the entire local decoder.
The global A7 `paper_fce` loss backpropagates through the shared image and text
towers; the local category FCE loss backpropagates through the same visual and
text towers plus the local decoder. The first training batch checks that both
losses reach both CLIP towers. To resume, keep the same E1-B initialization and
pass the E1-D latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage full \
  --init-checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_best.pth \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D/checkpoint_last.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D
```

### E1-E — direct pretrained full-joint training

If E1-A is still running and the goal is to jointly train all of CLIP and the
local decoder without waiting for E1-A/B, use E1-E. It loads OpenAI pretrained
CLIP B/16, randomly initializes the local decoder, then trains both CLIP towers
and the complete decoder from the first epoch with the same global/local
objectives. It has an independent output directory and can run without any E1
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage full \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E
```

To resume E1-E, use its latest checkpoint and keep the same output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage full \
  --resume outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E/checkpoint_last.pth \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E
```

### E1-0 — pretrained frozen B/16 control

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py train \
  --stage control \
  --config examples/task2/sample_code_submission/attrivision/configs/e1_shared_clip_b16.json \
  --data-root data \
  --device cuda:0 \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-0
```

E1-0 trains the same frozen-encoder local decoder as E1-B, initialized from
pretrained B/16 rather than E1-A. It writes the same three evaluation views.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-0 \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-0/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

Standalone E1-B evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-B \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

Standalone E1-C evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-C \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

Standalone E1-D evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-D \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

Standalone E1-E evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/task2/sample_code_submission/attrivision/shared_clip_b16/run.py eval \
  --experiment E1-E \
  --checkpoint outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E/checkpoint_best.pth \
  --data-root data \
  --device cuda:0
```

### Comparison CSV and Markdown

This uses the existing A7-B/32 validation result and each generated E1 result.
The script records descriptor-level mADM values and marks comparisons as
comparable only when the saved dataset root, validation protocol, split, and
category temperature agree.

```bash
python examples/task2/sample_code_submission/attrivision/shared_clip_b16/compare.py \
  --a7-b32 outputs/attrivision_ablation_mADM/A7/validation_metrics.json \
  --e1-0 outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-0/evaluation_val.json \
  --e1-a outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/evaluation_val.json \
  --e1-b outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/evaluation_val.json \
  --e1-c outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C/evaluation_val.json \
  --e1-d outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D/evaluation_val.json \
  --e1-e outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E/evaluation_val.json \
  --output-dir outputs/BackboneUpgrade/E1_SharedCLIP_B16/Comparison
```

The outputs are `comparison.csv`, `deltas.csv`, and `comparison.md`. E1-E can
run independently from pretrained CLIP; the staged path is E1-A → E1-B → E1-C
or E1-D.
