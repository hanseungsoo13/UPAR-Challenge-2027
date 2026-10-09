"""UPAR image dataset and stochastic positive-attribute text construction."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from upar.data import AnnotationTable, resolve_image_path

from .attribute_prompts import (
    CategoryPromptMapper, MixedCategoryPromptMapper, PaperAttributePromptMapper,
    prompts_for_attributes,
)


class AttriVisionDataset(Dataset):
    def __init__(self, table: AnnotationTable, roots: Sequence[Path], transform: Any,
                 max_samples: int | None = None) -> None:
        count = len(table.image_paths) if max_samples is None else min(max_samples, len(table.image_paths))
        self.image_paths = table.image_paths[:count]
        self.labels = table.labels[:count]
        self.roots = list(roots)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = resolve_image_path(self.image_paths[index], self.roots)
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.labels[index].copy())


@dataclass
class PromptBatch:
    images: torch.Tensor
    labels: torch.Tensor
    semantic_labels: torch.Tensor
    tokens: torch.Tensor
    selected_semantics: torch.Tensor
    text_owners: torch.Tensor


def semantic_label_matrix(
    labels: torch.Tensor, attribute_names: Sequence[str], prompt_mode: str,
) -> torch.Tensor:
    """Convert numeric UPAR labels to the text states used by a prompt mode."""
    if prompt_mode == "category_complete":
        return CategoryPromptMapper(attribute_names).encode(labels)
    if prompt_mode == "mixed_category":
        return MixedCategoryPromptMapper(attribute_names).encode(labels)
    if prompt_mode == "binary_positive":
        return labels > 0.5
    if prompt_mode == "paper_binary":
        return PaperAttributePromptMapper(attribute_names).encode(labels)
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def _unique_prompt_selection(
    semantic_labels: torch.Tensor,
    counts: Sequence[int],
    frequencies: torch.Tensor,
) -> list[list[int]] | None:
    """Greedily assign globally unique, rare-first text states to image slots."""
    used: set[int] = set()
    selected: list[list[int]] = []
    for row, count in zip(semantic_labels, counts):
        candidates = torch.nonzero(row, as_tuple=False).flatten().tolist()
        candidates.sort(key=lambda index: (float(frequencies[index]), index))
        available = [index for index in candidates if index not in used]
        if len(available) < count:
            return None
        choices = available[:count]
        selected.append(choices)
        used.update(choices)
    return selected


class UniquePromptBatchSampler(Sampler[list[int]]):
    """Shuffle images into batches admitting non-repeated sampled prompts.

    Section 4.2 of AttriVision explicitly avoids repeated image/label text pairs
    inside a contrastive batch. Batches may be shorter than ``batch_size`` when
    adding the next image would force a prompt collision; every image is still
    yielded exactly once per epoch.
    """

    def __init__(self, semantic_labels: torch.Tensor, batch_size: int,
                 prompts_per_image: int, seed: int) -> None:
        if semantic_labels.ndim != 2 or not semantic_labels.any(dim=1).all():
            raise ValueError("Every image must have at least one semantic prompt")
        if batch_size <= 0 or prompts_per_image <= 0:
            raise ValueError("batch size and prompts per image must be positive")
        self.semantic_labels = semantic_labels.bool().cpu()
        self.batch_size = batch_size
        self.prompts_per_image = prompts_per_image
        self.seed = seed
        self.epoch = 0
        self.frequencies = self.semantic_labels.sum(dim=0)
        priority = sorted(
            range(self.semantic_labels.shape[1]),
            key=lambda index: (int(self.frequencies[index]), index),
        )
        self._priority = np.asarray(priority, dtype=np.int64)
        self._priority_labels = self.semantic_labels[:, priority].numpy()

    def _build_batches(self, epoch: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        batches: list[list[int]] = []
        batch: list[int] = []
        used: set[int] = set()
        for index in torch.randperm(len(self.semantic_labels), generator=generator).tolist():
            candidates = self._priority[np.flatnonzero(self._priority_labels[index])].tolist()
            count = min(self.prompts_per_image, len(candidates))
            choices = [state for state in candidates if state not in used][:count]
            if len(batch) >= self.batch_size or len(choices) < count:
                if batch:
                    batches.append(batch)
                batch = [index]
                used = set(candidates[:count])
            else:
                batch.append(index)
                used.update(choices)
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches(self.epoch)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches(self.epoch))


class PromptCollator:
    """Sample one or several true semantic-state phrases for every image."""

    def __init__(self, attribute_names: Sequence[str], tokenizer: Any,
                 text_sampling: str = "single", multi_attributes: int = 3,
                 prompt_mode: str = "category_complete",
                 unique_prompts: bool = False,
                 semantic_frequencies: torch.Tensor | None = None,
                 stochastic: bool = True) -> None:
        if text_sampling not in {"single", "multi"}:
            raise ValueError("text_sampling must be 'single' or 'multi'")
        if multi_attributes <= 0:
            raise ValueError("multi_attributes must be positive")
        if prompt_mode not in {
            "category_complete", "mixed_category", "binary_positive", "paper_binary",
        }:
            raise ValueError("unknown prompt_mode")
        if prompt_mode == "category_complete":
            self.mapper = CategoryPromptMapper(attribute_names)
        elif prompt_mode == "mixed_category":
            self.mapper = MixedCategoryPromptMapper(attribute_names)
        elif prompt_mode == "paper_binary":
            self.mapper = PaperAttributePromptMapper(attribute_names)
        else:
            self.mapper = None
        self.prompts = (
            self.mapper.prompts
            if self.mapper is not None else prompts_for_attributes(attribute_names)
        )
        self.tokenizer = tokenizer
        self.text_sampling = text_sampling
        self.multi_attributes = multi_attributes
        self.prompt_mode = prompt_mode
        self.unique_prompts = unique_prompts
        self.semantic_frequencies = (
            semantic_frequencies.clone().cpu()
            if semantic_frequencies is not None else None
        )
        self.stochastic = stochastic

    def __call__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]) -> PromptBatch:
        images = torch.stack([sample[0] for sample in samples])
        labels = torch.stack([sample[1] for sample in samples])
        semantic_labels = self.mapper.encode(labels) if self.mapper is not None else labels.bool()
        counts = [
            1 if self.text_sampling == "single" else min(self.multi_attributes, int(row.sum()))
            for row in semantic_labels
        ]
        if self.unique_prompts:
            if self.semantic_frequencies is None:
                raise ValueError("unique prompt sampling requires semantic frequencies")
            selected = _unique_prompt_selection(
                semantic_labels, counts, self.semantic_frequencies,
            )
            if selected is None:
                raise RuntimeError("Batch contains unavoidable prompt collisions")
        else:
            selected = []
            for semantic_label, count in zip(semantic_labels, counts):
                positives = torch.nonzero(semantic_label, as_tuple=False).flatten().tolist()
                if not positives:
                    raise ValueError("Every training image needs at least one semantic state")
                selected.append(
                    random.sample(positives, count) if self.stochastic else positives[:count]
                )

        texts: list[str] = []
        owners: list[int] = []
        attribute_indices: list[int] = []
        for image_index, indices in enumerate(selected):
            for attribute_index in indices:
                texts.append(self.prompts[attribute_index])
                owners.append(image_index)
                attribute_indices.append(attribute_index)

        # Keep every selected phrase as an independent CLIP text input. For
        # multi sampling this produces T=sum_i K_i texts rather than averaging
        # K_i embeddings into one descriptor per image.
        tokens = self.tokenizer(texts)
        selected_semantics = torch.zeros(
            len(texts), semantic_labels.shape[1], dtype=torch.bool,
        )
        selected_semantics[torch.arange(len(texts)), attribute_indices] = True
        text_owners = torch.tensor(owners, dtype=torch.long)
        return PromptBatch(images, labels, semantic_labels, tokens, selected_semantics, text_owners)
