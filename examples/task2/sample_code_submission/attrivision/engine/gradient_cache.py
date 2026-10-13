"""Single-device gradient caching for global FCE and shared local FCE.

Cache representations without activation graphs, differentiate the complete
logical batch loss, then replay each encoder chunk with its original RNG state.
Representation gradients are already AMP-scaled; replay must not scale again.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CacheTask:
    encode: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]]
    inputs: Sequence[torch.Tensor]
    trainable: bool = True


@dataclass
class CacheResult:
    metrics: dict[str, float]
    gradient_norms: dict[str, float]


@dataclass
class _RandomState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None

    @classmethod
    def capture(cls, device: torch.device) -> _RandomState:
        return cls(
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )

    @contextmanager
    def replay(self, device: torch.device) -> Iterator[None]:
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda, device)
            yield


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(
        device_type=device.type, dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    )


def _outputs(value: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (value,) if torch.is_tensor(value) else tuple(value)


def _style_view(images: torch.Tensor) -> torch.Tensor:
    """Match E1-B's fixed exposure/contrast view without changing geometry."""
    mean = images.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
    std = images.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    rgb = ((rgb - 0.5) * 1.08 + 0.5) * 0.92
    return (rgb.clamp(0.0, 1.0) - mean) / std


def grouped_batches(loader: Iterable[Any], steps: int) -> Iterator[list[Any]]:
    if steps < 1:
        raise ValueError("grad-cache-steps must be at least 1")
    iterator = iter(loader)
    while group := list(islice(iterator, steps)):
        yield group


def merge_prompt_metadata(
    batches: Sequence[Any], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offset text owners by image count, including variable numbers of texts."""
    owners: list[torch.Tensor] = []
    image_offset = 0
    for batch in batches:
        owners.append(batch.text_owners + image_offset)
        image_offset += len(batch.images)
    return (
        torch.cat([batch.semantic_labels for batch in batches]).to(device),
        torch.cat([batch.selected_semantics for batch in batches]).to(device),
        torch.cat(owners).to(device),
    )


def cached_backward(
    tasks: Sequence[CacheTask],
    loss_fn: Callable[[list[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    scaler: Any,
    device: torch.device,
    amp: bool = False,
    diagnostics: dict[str, tuple[str, int, int]] | None = None,
) -> CacheResult:
    """Populate parameter gradients for one logical batch; do not step/zero them.

    The loss may directly use parameters such as CLIP logit_scale. Those receive
    their gradient once in the loss pass; encoder gradients come from replay.
    Encoder functions must be batch independent (CLIP/LayerNorm, not BatchNorm)
    and may use PyTorch CPU/CUDA randomness, which is restored for each replay.
    """
    representations: list[tuple[torch.Tensor, ...]] = []
    states: list[list[_RandomState]] = []
    sizes: list[list[int]] = []
    for task in tasks:
        chunks: list[tuple[torch.Tensor, ...]] = []
        task_states: list[_RandomState] = []
        task_sizes: list[int] = []
        if not task.inputs:
            raise ValueError("Each cache task requires at least one input chunk")
        for chunk in task.inputs:
            task_states.append(_RandomState.capture(device))
            with torch.no_grad(), _autocast(device, amp):
                values = _outputs(task.encode(chunk.to(device, non_blocking=True)))
            if not values or any(value.shape[0] != len(chunk) for value in values):
                raise ValueError("Cached encoder outputs must preserve the input batch dimension")
            if chunks and len(values) != len(chunks[0]):
                raise ValueError("Cached encoder output count changed between chunks")
            chunks.append(tuple(value.detach().float() for value in values))
            task_sizes.append(len(chunk))
        representations.append(tuple(
            torch.cat([chunk[index] for chunk in chunks]).detach().requires_grad_(task.trainable)
            for index in range(len(chunks[0]))
        ))
        states.append(task_states)
        sizes.append(task_sizes)

    with _autocast(device, amp):
        loss, metrics = loss_fn(representations)
    norms: dict[str, float] = {}
    for name, (loss_name, task_index, output_index) in (diagnostics or {}).items():
        feature = representations[task_index][output_index]
        gradient = torch.autograd.grad(
            metrics[loss_name], feature, retain_graph=True, allow_unused=True,
        )[0]
        norms[name] = float(gradient.detach().float().norm()) if gradient is not None else 0.0
        if norms[name] <= 0.0:
            raise RuntimeError(f"Gradient Cache loss is disconnected from {name}")
    scaler.scale(loss).backward()
    measured = {name: float(value.detach()) for name, value in metrics.items()}

    for task, features, rng_states, chunk_sizes in zip(tasks, representations, states, sizes):
        if not task.trainable:
            continue
        gradients = [feature.grad for feature in features]
        offset = 0
        for chunk, state, size in zip(task.inputs, rng_states, chunk_sizes):
            with state.replay(device), _autocast(device, amp):
                values = _outputs(task.encode(chunk.to(device, non_blocking=True)))
                terms = [
                    (value.float() * gradient[offset:offset + size]).sum()
                    for value, gradient in zip(values, gradients)
                    if gradient is not None
                ]
                if terms:
                    # Cached gradients already contain GradScaler's scale.
                    sum(terms).backward()
            offset += size
    return CacheResult(measured, norms)


def train_cached_epoch(
    model: Any, loader: Iterable[Any], optimizer: Any, scheduler: Any,
    scaler: Any, device: torch.device, amp: bool, grad_clip: float,
    steps: int, chunk_size: int, global_criterion: Any,
    optimizer_step: Callable[[Any, Any, Any], None], *,
    local_criterion: Any = None, state_tokens: torch.Tensor | None = None,
    state_features: torch.Tensor | None = None, anchor_model: Any = None,
    global_weight: float = 1.0, local_weight: float = 0.25,
    anchor_weight: float = 0.0,
    consistency_variant: str = "B0", feature_consistency_weight: float = 0.0,
    attention_consistency_weight: float = 0.0,
) -> dict[str, float]:
    """One epoch for A7 global FCE, or shared vision/local C and full-joint D."""
    model.train()
    shared = local_criterion is not None
    full = shared and model.stage == "full"
    frozen_local = shared and model.stage == "local"
    if consistency_variant not in {"B0", "B1", "B2"}:
        raise ValueError("consistency_variant must be B0, B1, or B2")
    if consistency_variant != "B0" and not frozen_local:
        raise ValueError("E1-B consistency variants require a frozen local-query stage")
    if consistency_variant == "B0" and (feature_consistency_weight or attention_consistency_weight):
        raise ValueError("B0 requires zero consistency weights")
    if consistency_variant == "B1" and (feature_consistency_weight <= 0 or attention_consistency_weight):
        raise ValueError("B1 requires a positive feature weight and zero attention weight")
    if consistency_variant == "B2" and (feature_consistency_weight <= 0 or attention_consistency_weight <= 0):
        raise ValueError("B2 requires positive feature and attention consistency weights")
    base = model.base if shared else model
    text_trainable = any(
        parameter.requires_grad for name, parameter in base.clip.named_parameters()
        if not name.startswith("visual.") and name != "logit_scale"
    )
    if shared:
        model.assert_trainability()
        if state_tokens is None or (not full and state_features is None):
            raise ValueError("Shared Gradient Cache requires local text prototypes")
    totals: dict[str, float] = {}
    norms: dict[str, float] = {}
    samples = microbatches = groups = updates = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for group in grouped_batches(loader, steps):
        targets, selected, owners = merge_prompt_metadata(group, device)
        image_chunks = [batch.images for batch in group]
        token_chunks = [chunk for batch in group for chunk in batch.tokens.split(chunk_size)]
        if frozen_local:
            # The image tower stays frozen; replay only the trainable category
            # queries, cross-attention, FFN, normalization and projection.
            tasks = [CacheTask(
                lambda images: model.joint_features(images)[1], image_chunks,
            )]
            if consistency_variant in {"B1", "B2"}:
                def encode_consistency(images: torch.Tensor):
                    # The original objective disables decoder dropout for both
                    # style views. Restore train mode before replay completes.
                    model.eval()
                    try:
                        if consistency_variant == "B2":
                            _, original, _, original_attention = model.joint_features_with_tokens(images)
                            _, styled, _, styled_attention = model.joint_features_with_tokens(
                                _style_view(images),
                            )
                            return original, styled, original_attention, styled_attention
                        original = model.joint_features(images)[1]
                        styled = model.joint_features(_style_view(images))[1]
                        return original, styled
                    finally:
                        model.train()

                tasks.append(CacheTask(encode_consistency, image_chunks))
        else:
            tasks = [
                CacheTask(model.joint_features if shared else model.encode_image, image_chunks),
                CacheTask(model.encode_text, token_chunks, trainable=text_trainable),
            ]
            if full:
                tasks.append(CacheTask(model.encode_text, list(state_tokens.cpu().split(chunk_size))))
        anchor_features = None
        if anchor_model is not None:
            with torch.no_grad(), _autocast(device, amp):
                anchor_features = torch.cat([
                    anchor_model.base.encode_image(chunk.to(device)).float()
                    for chunk in image_chunks
                ])

        def loss_fn(reps: list[tuple[torch.Tensor, ...]]):
            anchor_loss = targets.new_zeros((), dtype=torch.float32)
            if frozen_local:
                image_features = None
                global_loss = anchor_loss
                local_features = reps[0][0]
            else:
                image_features, text_features = reps[0][0], reps[1][0]
                scale = base.logit_scale.exp().clamp(max=100.0)
                global_output = global_criterion(scale * image_features @ text_features.T, targets, selected, owners)
                global_loss = global_output.loss
                local_features = reps[0][1] if shared else None
            local_loss = anchor_loss
            feature_consistency = anchor_loss
            attention_consistency = anchor_loss
            if shared:
                prototypes = state_features if frozen_local else (reps[2][0] if full else state_features)
                valid = torch.stack([targets[:, indices].sum(dim=1) > 0 for indices in local_criterion.groups], dim=1)
                local_loss = local_criterion(local_features, prototypes, targets, valid).loss
            if consistency_variant in {"B1", "B2"}:
                feature_consistency = 1.0 - F.cosine_similarity(
                    reps[1][0].float(), reps[1][1].float(), dim=-1,
                ).mean()
                if consistency_variant == "B2":
                    attention_consistency = 1.0 - F.cosine_similarity(
                        reps[1][2].float(), reps[1][3].float(), dim=-1,
                    ).mean()
            if anchor_features is not None:
                anchor_loss = (1.0 - F.cosine_similarity(image_features.float(), anchor_features.float(), dim=-1)).mean()
            if frozen_local:
                total = (
                    local_loss
                    + feature_consistency_weight * feature_consistency
                    + attention_consistency_weight * attention_consistency
                )
            else:
                total = (
                    global_weight * global_loss
                    + local_weight * local_loss
                    + anchor_weight * anchor_loss
                )
            zero = local_loss.new_zeros(())
            return total, {
                "total_loss": total, "global_loss": global_loss, "local_loss": local_loss,
                "clip_feature_anchor_loss": anchor_loss,
                "feature_consistency_loss": feature_consistency,
                "attention_consistency_loss": attention_consistency,
                "i2t": zero if frozen_local else global_output.i2t,
                "t2i": zero if frozen_local else global_output.t2i,
            }

        diagnostics = None
        if groups == 0:
            diagnostics = {} if frozen_local else {"cache_global_image_gradient": ("global_loss", 0, 0)}
            if shared and not frozen_local:
                diagnostics["cache_local_image_gradient"] = ("local_loss", 0, 1)
            if anchor_model is not None:
                diagnostics["cache_anchor_image_gradient"] = ("clip_feature_anchor_loss", 0, 0)
            if text_trainable:
                diagnostics["cache_global_text_gradient"] = ("global_loss", 1, 0)
            if full:
                diagnostics["cache_local_text_gradient"] = ("local_loss", 2, 0)
        optimizer.zero_grad(set_to_none=True)
        result = cached_backward(tasks, loss_fn, scaler, device, amp, diagnostics)
        scaler.unscale_(optimizer)
        if groups == 0:
            if frozen_local:
                model.assert_local_gradients()
            elif not any(p.grad is not None for p in base.clip.visual.parameters() if p.requires_grad):
                raise RuntimeError("Gradient Cache replay produced no vision gradient")
            if text_trainable and not any(
                p.grad is not None for name, p in base.clip.named_parameters()
                if not name.startswith("visual.") and name != "logit_scale" and p.requires_grad
            ):
                raise RuntimeError("Gradient Cache replay produced no text gradient")
            if shared:
                if not frozen_local:
                    model.assert_local_gradients()
            norms.update(result.gradient_norms)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        previous_scale = float(scaler.get_scale())
        optimizer_step(scaler, optimizer, scheduler)
        updates += int(float(scaler.get_scale()) >= previous_scale)
        count = sum(len(batch.images) for batch in group)
        for name, value in result.metrics.items():
            totals[name] = totals.get(name, 0.0) + value * count
        samples += count
        microbatches += len(group)
        groups += 1
    metrics = {name: value / max(samples, 1) for name, value in totals.items()}
    metrics.update(norms)
    metrics.update({
        "train_batches": float(microbatches), "optimizer_steps": float(updates),
        "cache_groups": float(groups), "grad_cache_steps": float(steps),
        "average_batch_size": samples / max(microbatches, 1),
        "average_contrastive_batch_size": samples / max(groups, 1),
        "effective_batch_size": float(chunk_size * steps),
    })
    if device.type == "cuda":
        metrics["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        metrics["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    return metrics
