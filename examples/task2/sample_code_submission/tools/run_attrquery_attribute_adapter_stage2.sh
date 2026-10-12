#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> $0 <NoAdapter|PatchFiLM>" >&2
  echo "Runs one independent Stage 2 condition from the original AttrQuery LocalFCE checkpoint." >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

architecture="$1"
case "$architecture" in
  NoAdapter|PatchFiLM) ;;
  *)
    usage
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../../.." && pwd)"
cd "$repo_root"

exec python examples/task2/sample_code_submission/tools/train_attrquery_attribute_adapter.py \
  --architecture "$architecture" \
  --stage 2 \
  --a7-checkpoint outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth \
  --attrquery-checkpoint outputs/AttrQuery-FrozenCLIP-LocalFCE/checkpoint_best.pth \
  --output-dir "outputs/AttrQuery-AttributeAdapter/Stage2/$architecture" \
  --device cuda:0
