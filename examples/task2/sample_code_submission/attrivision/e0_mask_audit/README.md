# E0 FCE mask audit

This folder contains the no-training supervision audit for the AttriVision A1
formulation. It recreates train batches from the UPAR GT and calls the existing
PromptCollator and FocalCLIPLoss.positive_mask code. Images are replaced by
tiny dummy tensors, tokenization is shape-only, and no checkpoint or CLIP
encoder is loaded.

The default configuration matches A1 in run_attrivision_ablation.py:

- category_complete
- multi sampling with at most three semantic states per image
- multi_positive FCE target
- regular shuffled batches, no unique-prompt sampler

Run one full train epoch from the repository root:

~~~bash
python examples/task2/sample_code_submission/attrivision/e0_mask_audit/e0_mask_audit.py \
  --data-root data \
  --output-dir outputs/attrivision_e0_mask_audit
~~~

For the recommended multi-seed check:

~~~bash
python examples/task2/sample_code_submission/attrivision/e0_mask_audit/e0_mask_audit.py \
  --data-root data \
  --seeds 42 43 44 45 46
~~~

To expose the owner-only diagonal false negatives under the same sampler:

~~~bash
python examples/task2/sample_code_submission/attrivision/e0_mask_audit/e0_mask_audit.py \
  --data-root data \
  --contrastive-target diagonal \
  --seeds 42 43 44
~~~

Artifacts are written under the audit output directory:

- summary.json: configuration, per-seed results, and mean/std aggregate
- seed_<seed>.json: one complete result per seed
- seed_summary.csv: compact per-seed table
- batch_metrics.csv: candidate-pair counts for every observed batch

FNR/FPR denominators are restricted to candidate pairs present in the recreated
batch. The expected GT mask is an image/text positive when the image's 52-state
GT overlaps the sampled text's selected semantic state. The audit reports
image-to-text and text-to-image separately.

For category_complete, fallback contribution is reported both as a share of
sampled positive texts and as a share of GT-positive candidate pairs. Category
conflict is the rate of category/image slots with more than one active state;
the per-category breakdown is included in summary.json. This measures the
target geometry relevant to Category-CE, but A1 itself does not train with
Category-CE.
