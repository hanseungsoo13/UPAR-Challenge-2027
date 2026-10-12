# AttriVision / AttriQuery BackboneUpgrade

This directory adds isolated E1/E2/E3 experiments. It does not modify the A7,
AttriQuery, or prior experiment code or checkpoints. Run commands from the
`UPAR-Challenge-2027` repository root.

## Repository findings reflected in the implementation

- A7 is OpenCLIP `ViT-B-32-quickgelu` with OpenAI initialization, 224×224 input,
  and a 512-dimensional normalized image/text space. The wrapper contains a
  40-bit binary-head module, while this A7 run uses `paper_fce` and sets
  `binary_head_loss=false`; its retrieval descriptor comes from text prototypes.
- The A7 checkpoint used by E1/E2 is
  `outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth`. The saved
  validation result records epoch 8, mADM 0.68627, mAP 0.30377, and Rank-1
  0.32611. A7 uses the 40-bit annotation mapped to 52 semantic states in 12
  categories, including the mixed/unknown complement states. Validation
  ranking uses Native52 Category-NLL at temperature 0.01; best A7 selection is
  by validation mADM.
- Existing `AttrQuery-FrozenCLIP-LocalFCE` freezes the A7 OpenCLIP ViT-B/32,
  extracts its final transformer sequence with a temporary hook, removes CLS,
  and trains category queries, cross-attention, FFN, and an output projection
  with category-local focal cross-entropy. It uses the A7 text encoder's 52
  cached prototypes. Its saved latest result is epoch 11 with mADM 0.68374.
- A7's `paper_fce` is symmetric image-to-text/text-to-image focal cross-entropy
  over normalized image/text similarities and a multi-positive target. Its
  Native52 scorer computes category softmax at temperature 0.01 and then
  category NLL. The current frozen AttrQuery local loss compares each of the
  12 category features only with that category's states, uses temperature
  0.01 and focal gamma 2, and ignores categories without a valid target.
- Existing joint AttriQuery uses a local/global fusion of category
  log-probabilities, with alpha 0.5 used by its protocol check. It evaluates
  validation mADM, mAP, Rank-1/5/10, and mINP. The new experiments retain the
  same 12-category decoder structure and score fusion rule; E1/E2 replace only
  the local representation.
- The best saved AttriQuery comparison reference is
  `AttrQuery-PartialJointFinetune/FrozenBackbone-Control`: frozen A7 vision,
  trained category-local branch, alpha 0.5, best validation mADM 0.70271 at
  epoch 6 (last saved epoch 8). Its best result is read from the recorded
  history and selection metadata. The separate `AttrQuery-FrozenCLIP-LocalFCE`
  local-only run's latest saved row is epoch 11, mADM 0.68374; it is not the
  repository's strongest AttriQuery result.
- `NoAdapter`, adapter-only, and joint-training settings exist as separate
  prior experiments (`AttrQuery-FrozenCLIP-LocalFCE`, `AttrQuery-PartialJointFinetune`,
  and `AttrQuery-FromScratchJointFCE`). E1/E2 use a frozen external local
  backbone plus trainable projector/query/decoder. E3-A freezes both SigLIP2
  encoders; E3-B optionally tunes the last two vision blocks and uses a
  SigLIP2-compatible sigmoid focal objective.

## Model choices

- **E1:** A7 frozen global branch; frozen `openai/clip-vit-base-patch16` local
  branch. CLS is removed and only the 196 spatial tokens reach the local
  projector and existing category cross-attention decoder.
- **E2:** A7 frozen global branch; frozen DINOv3 ViT-B/16 local branch. The
  implementation removes one CLS plus the configured register tokens before
  local cross-attention. It uses the DINO image processor's normalization.
- **E3-A:** SigLIP2 ViT-B/16 supplies both global and local features. Its pooled
  global representation and projected text representations are L2-normalized
  and scored with the pretrained SigLIP2 scale and bias. The global adapter
  preserves SigLIP2's sigmoid alignment, then normalizes those confidences
  within each of the 12 semantic categories for Native52 Category-NLL. Its
  spatial patch sequence is separate from the pooled global representation.
- **E3-B:** Starts from E3-A's validation-selected best checkpoint by default,
  keeps the text encoder frozen, and trains the last two vision blocks,
  projector, query, and decoder. Its global attribute supervision is a
  sigmoid focal binary objective over the same 52-state ontology; it does not
  port A7's CLIP softmax FCE blindly. The existing category-local FCE remains
  on the local branch with weight 0.25.
- All four configurations use the same seed, 224 input, 30-epoch target,
  batch size 128, A7 resize/pad/crop geometry, AdamW schedule, category
  temperature, and alpha 0.5 fusion. Every epoch evaluates global-only,
  local-only, and fused scores. Checkpoint selection uses fused validation
  mADM. E3's two-branch replacement is a full-backbone experiment rather than
  a controlled local-only comparison.
- Batch size 128 is the shared default, not a measured memory guarantee. The
  frozen A7 plus a second ViT raises E1/E2 inference memory; E3-B has the
  largest training activation footprint because it backpropagates through two
  SigLIP2 vision blocks. Training accepts `--batch-size`, `--eval-batch-size`,
  `--num-workers`, and `--device` overrides; evaluation accepts the latter
  three. Reduce the batch sizes if the available GPU cannot hold the defaults.
  Exact peak memory is written to the training metrics and requires a real run.

The dataset returns unnormalized RGB tensors after the shared geometric
augmentation. Each encoder applies its own processor mean and standard
deviation. Frozen encoders are reconstructed from their pretrained source at
evaluation; experiment checkpoints store trainable weights and the cached
52-state prompts. E1/E2 evaluation also requires the original A7 checkpoint.
Checkpoint metadata records the config, checkpoint path, model ID, epoch,
feature dimensions, patch count, parameters, and evaluation metrics.

## Dependencies and model access

The checked-in `environment.yml` pins Python/PyTorch but does not specify a
Transformers version. DINOv3 support first appeared in released Transformers
4.56.0, so install the optional requirements before running these scripts:

```bash
pip install -r requirements-backbone-upgrade.txt
```

This code uses the documented Transformers `CLIPVisionModel`, `AutoModel`,
`AutoImageProcessor`, `AutoProcessor`, and SigLIP2 text/vision interfaces. No
package was installed and no model was loaded while implementing this change.

DINOv3's Hugging Face repository is gated. Before E2, sign in to Hugging Face,
request and accept the DINOv3 model terms, wait for access approval, then
authenticate the environment used for the run. This code does not download
weights until the user launches an experiment. See the
[DINOv3 model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and [Meta's DINOv3 download FAQ](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/discussions/6).

## Output layout

Each experiment writes its own `config.json`, `model_configuration.json`,
`parameter_counts.json`, `train.log`, `metrics.csv`, `metrics_latest.json`,
`checkpoint_best.pth`, and `checkpoint_last.pth` below
`outputs/BackboneUpgrade/<experiment>/`. Evaluation writes
`evaluation_val.json`; optional gallery caches are saved only when requested.
Comparison writes `comparison.csv` and `comparison.md` to
`outputs/BackboneUpgrade/Comparison/`.

E3-B defaults to requiring
`outputs/BackboneUpgrade/E3A_SigLIP2_Frozen/checkpoint_best.pth`. To explicitly
start E3-B from pretrained SigLIP2 instead, pass `--init-source pretrained` to
the training command. Smoke checks also accept `--pretrained-init` for this
case. Smoke checkpoints are temporary and carry a marker that evaluation
rejects.

See the final handoff for the four experiment-specific smoke, train, validation,
best-checkpoint evaluation, and comparison commands.
