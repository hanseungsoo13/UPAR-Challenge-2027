"""Natural-language prompts for the 40 canonical UPAR attributes."""
from __future__ import annotations

from collections.abc import Sequence

import torch


ATTRIBUTE_PROMPTS: dict[str, str] = {
    "Age-Young": "a photo of a young person",
    "Age-Adult": "a photo of an adult person",
    "Age-Old": "a photo of an elderly person",
    "Gender-Female": "a photo of a woman",
    "Hair-Length-Short": "a photo of a person with short hair",
    "Hair-Length-Long": "a photo of a person with long hair",
    "Hair-Length-Bald": "a photo of a bald person",
    "UpperBody-Length-Short": "a photo of a person wearing short sleeves",
    "UpperBody-Color-Black": "a photo of a person wearing a black upper-body garment",
    "UpperBody-Color-Blue": "a photo of a person wearing a blue upper-body garment",
    "UpperBody-Color-Brown": "a photo of a person wearing a brown upper-body garment",
    "UpperBody-Color-Green": "a photo of a person wearing a green upper-body garment",
    "UpperBody-Color-Grey": "a photo of a person wearing a grey upper-body garment",
    "UpperBody-Color-Orange": "a photo of a person wearing an orange upper-body garment",
    "UpperBody-Color-Pink": "a photo of a person wearing a pink upper-body garment",
    "UpperBody-Color-Purple": "a photo of a person wearing a purple upper-body garment",
    "UpperBody-Color-Red": "a photo of a person wearing a red upper-body garment",
    "UpperBody-Color-White": "a photo of a person wearing a white upper-body garment",
    "UpperBody-Color-Yellow": "a photo of a person wearing a yellow upper-body garment",
    "UpperBody-Color-Other": "a photo of a person wearing another upper-body color",
    "LowerBody-Length-Short": "a photo of a person wearing short lower-body clothing",
    "LowerBody-Color-Black": "a photo of a person wearing black lower-body clothing",
    "LowerBody-Color-Blue": "a photo of a person wearing blue lower-body clothing",
    "LowerBody-Color-Brown": "a photo of a person wearing brown lower-body clothing",
    "LowerBody-Color-Green": "a photo of a person wearing green lower-body clothing",
    "LowerBody-Color-Grey": "a photo of a person wearing grey lower-body clothing",
    "LowerBody-Color-Orange": "a photo of a person wearing orange lower-body clothing",
    "LowerBody-Color-Pink": "a photo of a person wearing pink lower-body clothing",
    "LowerBody-Color-Purple": "a photo of a person wearing purple lower-body clothing",
    "LowerBody-Color-Red": "a photo of a person wearing red lower-body clothing",
    "LowerBody-Color-White": "a photo of a person wearing white lower-body clothing",
    "LowerBody-Color-Yellow": "a photo of a person wearing yellow lower-body clothing",
    "LowerBody-Color-Other": "a photo of a person wearing another lower-body color",
    "LowerBody-Type-Trousers&Shorts": "a photo of a person wearing trousers or shorts",
    "LowerBody-Type-Skirt&Dress": "a photo of a person wearing a skirt or dress",
    "Accessory-Backpack": "a photo of a person carrying a backpack",
    "Accessory-Bag": "a photo of a person carrying a bag",
    "Accessory-Glasses-Normal": "a photo of a person wearing eyeglasses",
    "Accessory-Glasses-Sun": "a photo of a person wearing sunglasses",
    "Accessory-Hat": "a photo of a person wearing a hat",
}

# Binary counter-prompts used to turn CLIP similarities into one probability
# per official UPAR attribute.  Multi-class fields (age, hair, colours, ...)
# are intentionally treated as 40 independent binary decisions here because
# Task 2 queries are 40-bit vectors and the official distance is defined in
# that space.
NEGATIVE_ATTRIBUTE_PROMPTS: dict[str, str] = {
    "Age-Young": "a photo of a person who is not young",
    "Age-Adult": "a photo of a person who is not an adult",
    "Age-Old": "a photo of a person who is not elderly",
    "Gender-Female": "a photo of a man",
    "Hair-Length-Short": "a photo of a person without short hair",
    "Hair-Length-Long": "a photo of a person without long hair",
    "Hair-Length-Bald": "a photo of a person who is not bald",
    "UpperBody-Length-Short": "a photo of a person wearing long sleeves",
    "LowerBody-Length-Short": "a photo of a person wearing long lower-body clothing",
    "LowerBody-Type-Trousers&Shorts": "a photo of a person not wearing trousers or shorts",
    "LowerBody-Type-Skirt&Dress": "a photo of a person not wearing a skirt or dress",
    "Accessory-Backpack": "a photo of a person without a backpack",
    "Accessory-Bag": "a photo of a person without a bag",
    "Accessory-Glasses-Normal": "a photo of a person without eyeglasses",
    "Accessory-Glasses-Sun": "a photo of a person without sunglasses",
    "Accessory-Hat": "a photo of a person without a hat",
}

for _body, _garment in (
    ("UpperBody", "upper-body garment"),
    ("LowerBody", "lower-body clothing"),
):
    for _color in (
        "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
        "Purple", "Red", "White", "Yellow", "Other",
    ):
        if _color == "Other":
            _negative = f"a photo of a person not wearing another {_garment} color"
        else:
            _negative = f"a photo of a person not wearing {_color.lower()} {_garment}"
        NEGATIVE_ATTRIBUTE_PROMPTS[f"{_body}-Color-{_color}"] = _negative


def prompts_for_attributes(attribute_names: Sequence[str]) -> list[str]:
    missing = [name for name in attribute_names if name not in ATTRIBUTE_PROMPTS]
    if missing:
        raise ValueError(f"No natural-language prompt defined for attributes: {missing}")
    return [ATTRIBUTE_PROMPTS[name] for name in attribute_names]


class PaperAttributePromptMapper:
    """Map each official binary attribute to its positive/negative prompt.

    The paper describes one natural-language state for both presence and
    absence of every attribute.  This deliberately keeps the official 40-bit
    vocabulary instead of introducing mutually-exclusive category states.
    State order is ``negative, positive`` for every attribute.
    """

    def __init__(self, attribute_names: Sequence[str]) -> None:
        negative, positive = prompt_pairs_for_attributes(attribute_names)
        self.attribute_names = list(attribute_names)
        self.prompts = [text for pair in zip(negative, positive) for text in pair]
        self.keys = [
            key
            for name in self.attribute_names
            for key in (f"{name}=0", f"{name}=1")
        ]

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attribute_names):
            raise ValueError(
                f"Expected labels [B,{len(self.attribute_names)}], got {tuple(labels.shape)}"
            )
        binary = labels > 0.5
        semantic = torch.zeros(
            labels.shape[0], len(self.prompts), dtype=torch.bool, device=labels.device,
        )
        columns = torch.arange(len(self.attribute_names), device=labels.device)
        semantic[:, 2 * columns] = ~binary
        semantic[:, 2 * columns + 1] = binary
        return semantic


def prompt_pairs_for_attributes(attribute_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return negative and positive prompts in the requested attribute order."""
    missing_positive = [name for name in attribute_names if name not in ATTRIBUTE_PROMPTS]
    missing_negative = [name for name in attribute_names if name not in NEGATIVE_ATTRIBUTE_PROMPTS]
    if missing_positive or missing_negative:
        raise ValueError(
            "No paired prompt defined for attributes: "
            f"positive={missing_positive}, negative={missing_negative}"
        )
    return (
        [NEGATIVE_ATTRIBUTE_PROMPTS[name] for name in attribute_names],
        [ATTRIBUTE_PROMPTS[name] for name in attribute_names],
    )


# AttriVision describes attributes as category values, including an explicit
# phrase for a category's negative/complement state. The keys below define a
# stable semantic vocabulary shared by training and Task 2 query encoding.
CATEGORY_PROMPTS: dict[str, str] = {
    "age_young": "a photo of a young person",
    "age_adult": "a photo of an adult person",
    "age_old": "a photo of an elderly person",
    "age_unknown": "a photo of a person of unspecified age",
    "gender_woman": "a photo of a woman",
    "gender_man": "a photo of a man",
    "hair_short": "a photo of a person with short hair",
    "hair_long": "a photo of a person with long hair",
    "hair_bald": "a photo of a bald person",
    "hair_other": "a photo of a person with another hairstyle",
    "upper_sleeves_short": "a photo of a person wearing short sleeves",
    "upper_sleeves_long": "a photo of a person wearing long sleeves",
    "lower_length_short": "a photo of a person wearing short lower-body clothing",
    "lower_length_long": "a photo of a person wearing long lower-body clothing",
    "lower_trousers_shorts": "a photo of a person wearing trousers or shorts",
    "lower_skirt_dress": "a photo of a person wearing a skirt or dress",
    "lower_type_other": "a photo of a person wearing another type of lower-body garment",
    "backpack_yes": "a photo of a person carrying a backpack",
    "backpack_no": "a photo of a person without a backpack",
    "bag_yes": "a photo of a person carrying a bag",
    "bag_no": "a photo of a person without a bag",
    "glasses_normal": "a photo of a person wearing eyeglasses",
    "glasses_sun": "a photo of a person wearing sunglasses",
    "glasses_none": "a photo of a person without glasses",
    "hat_yes": "a photo of a person wearing a hat",
    "hat_no": "a photo of a person without a hat",
}

for _prefix, _garment in (("upper", "upper-body garment"), ("lower", "lower-body clothing")):
    for _color in (
        "black", "blue", "brown", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow", "other",
    ):
        if _color == "other":
            _phrase = f"another {_garment} color"
        else:
            _article = "an" if _color == "orange" else "a"
            _phrase = f"{_article} {_color} {_garment}"
        CATEGORY_PROMPTS[f"{_prefix}_color_{_color}"] = f"a photo of a person wearing {_phrase}"
    CATEGORY_PROMPTS[f"{_prefix}_color_unspecified"] = (
        f"a photo of a person wearing an unspecified-color {_garment}"
    )


class CategoryPromptMapper:
    """Map UPAR binary rows to complete semantic states for 12 categories."""

    _MULTI_GROUPS = (
        ("age", ("Age-Young", "Age-Adult", "Age-Old"),
         ("age_young", "age_adult", "age_old"), "age_unknown"),
        ("hair", ("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"),
         ("hair_short", "hair_long", "hair_bald"), "hair_other"),
        ("upper_color", tuple(f"UpperBody-Color-{color}" for color in (
            "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
            "Purple", "Red", "White", "Yellow", "Other",
        )), tuple(f"upper_color_{color}" for color in (
            "black", "blue", "brown", "green", "grey", "orange", "pink",
            "purple", "red", "white", "yellow", "other",
        )), "upper_color_unspecified"),
        ("lower_color", tuple(f"LowerBody-Color-{color}" for color in (
            "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
            "Purple", "Red", "White", "Yellow", "Other",
        )), tuple(f"lower_color_{color}" for color in (
            "black", "blue", "brown", "green", "grey", "orange", "pink",
            "purple", "red", "white", "yellow", "other",
        )), "lower_color_unspecified"),
        ("lower_type", ("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"),
         ("lower_trousers_shorts", "lower_skirt_dress"), "lower_type_other"),
        ("glasses", ("Accessory-Glasses-Normal", "Accessory-Glasses-Sun"),
         ("glasses_normal", "glasses_sun"), "glasses_none"),
    )
    _BINARY_GROUPS = (
        ("Gender-Female", "gender_woman", "gender_man"),
        ("UpperBody-Length-Short", "upper_sleeves_short", "upper_sleeves_long"),
        ("LowerBody-Length-Short", "lower_length_short", "lower_length_long"),
        ("Accessory-Backpack", "backpack_yes", "backpack_no"),
        ("Accessory-Bag", "bag_yes", "bag_no"),
        ("Accessory-Hat", "hat_yes", "hat_no"),
    )

    def __init__(self, attribute_names: Sequence[str]) -> None:
        if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
            raise ValueError("Category prompt mapping requires 40 unique UPAR attributes")
        attribute_index = {name: index for index, name in enumerate(attribute_names)}
        required = {
            column
            for _, columns, _, _ in self._MULTI_GROUPS
            for column in columns
        } | {column for column, _, _ in self._BINARY_GROUPS}
        missing = sorted(required - attribute_index.keys())
        if missing:
            raise ValueError(f"Category prompt mapping is missing attributes: {missing}")

        keys: list[str] = []
        for _, _, state_keys, fallback_key in self._MULTI_GROUPS:
            keys.extend(state_keys)
            keys.append(fallback_key)
        for _, positive_key, negative_key in self._BINARY_GROUPS:
            keys.extend((positive_key, negative_key))
        if len(keys) != len(set(keys)) or set(keys) != set(CATEGORY_PROMPTS):
            raise RuntimeError("Category prompt vocabulary definition is inconsistent")

        self.attribute_names = list(attribute_names)
        self.keys = keys
        self.prompts = [CATEGORY_PROMPTS[key] for key in keys]
        self._attribute_index = attribute_index
        self._semantic_index = {key: index for index, key in enumerate(keys)}

    def category_indices(self) -> list[list[int]]:
        """Return the canonical 12-category partition of the 52 state indices."""
        groups: list[list[int]] = []
        for _, _, state_keys, fallback_key in self._MULTI_GROUPS:
            groups.append([self._semantic_index[key] for key in (*state_keys, fallback_key)])
        for _, positive_key, negative_key in self._BINARY_GROUPS:
            groups.append([
                self._semantic_index[positive_key], self._semantic_index[negative_key],
            ])
        covered = [index for group in groups for index in group]
        if len(groups) != 12 or len(covered) != 52 or len(set(covered)) != 52:
            raise RuntimeError("Category groups must partition all 52 semantic states")
        return groups

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attribute_names):
            raise ValueError(f"Expected labels [B,40], got {tuple(labels.shape)}")
        binary = labels > 0.5
        semantic = torch.zeros(
            labels.shape[0], len(self.keys), dtype=torch.bool, device=labels.device,
        )
        for _, columns, state_keys, fallback_key in self._MULTI_GROUPS:
            columns_tensor = torch.tensor(
                [self._attribute_index[column] for column in columns], device=labels.device,
            )
            values = binary[:, columns_tensor]
            for offset, state_key in enumerate(state_keys):
                semantic[:, self._semantic_index[state_key]] = values[:, offset]
            semantic[:, self._semantic_index[fallback_key]] = ~values.any(dim=1)
        for column, positive_key, negative_key in self._BINARY_GROUPS:
            values = binary[:, self._attribute_index[column]]
            semantic[:, self._semantic_index[positive_key]] = values
            semantic[:, self._semantic_index[negative_key]] = ~values
        return semantic


class MixedCategoryPromptMapper:
    """Map UPAR rows to the Task-2 mixed categorical/multi-label vocabulary.

    The legacy :class:`CategoryPromptMapper` deliberately exposes 52 states and
    applies a category-local softmax to every group.  Task 2 annotations show
    genuine multi-positive rows for hair, both colour groups, and lower-body
    type, so A7-mixed keeps those groups as sigmoid/BCE groups while retaining
    categorical competition for the remaining groups.

    The mixed vocabulary has 50 states: the two colour ``unspecified``
    fallbacks are removed because they are never positive in the Task-2 train
    or validation annotations.  ``age_unknown``, ``hair_other``,
    ``lower_type_other``, and ``glasses_none`` remain as data-backed fallback
    states.
    """

    _SINGLE_GROUPS = (
        ("age", ("Age-Young", "Age-Adult", "Age-Old"),
         ("age_young", "age_adult", "age_old"), "age_unknown"),
        ("glasses", ("Accessory-Glasses-Normal", "Accessory-Glasses-Sun"),
         ("glasses_normal", "glasses_sun"), "glasses_none"),
    )
    _MULTILABEL_GROUPS = (
        ("hair", ("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"),
         ("hair_short", "hair_long", "hair_bald"), "hair_other"),
        ("upper_color", tuple(f"UpperBody-Color-{color}" for color in (
            "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
            "Purple", "Red", "White", "Yellow", "Other",
        )), tuple(f"upper_color_{color}" for color in (
            "black", "blue", "brown", "green", "grey", "orange", "pink",
            "purple", "red", "white", "yellow", "other",
        )), None),
        ("lower_color", tuple(f"LowerBody-Color-{color}" for color in (
            "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
            "Purple", "Red", "White", "Yellow", "Other",
        )), tuple(f"lower_color_{color}" for color in (
            "black", "blue", "brown", "green", "grey", "orange", "pink",
            "purple", "red", "white", "yellow", "other",
        )), None),
        ("lower_type", ("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"),
         ("lower_trousers_shorts", "lower_skirt_dress"), "lower_type_other"),
    )
    _BINARY_GROUPS = (
        ("Gender-Female", "gender_woman", "gender_man"),
        ("UpperBody-Length-Short", "upper_sleeves_short", "upper_sleeves_long"),
        ("LowerBody-Length-Short", "lower_length_short", "lower_length_long"),
        ("Accessory-Backpack", "backpack_yes", "backpack_no"),
        ("Accessory-Bag", "bag_yes", "bag_no"),
        ("Accessory-Hat", "hat_yes", "hat_no"),
    )

    def __init__(self, attribute_names: Sequence[str]) -> None:
        if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
            raise ValueError("Mixed category mapping requires 40 unique UPAR attributes")
        self.attribute_names = list(attribute_names)
        self._attribute_index = {name: index for index, name in enumerate(attribute_names)}
        required = {
            column
            for _, columns, _, _ in (*self._SINGLE_GROUPS, *self._MULTILABEL_GROUPS)
            for column in columns
        } | {column for column, _, _ in self._BINARY_GROUPS}
        missing = sorted(required - self._attribute_index.keys())
        if missing:
            raise ValueError(f"Mixed category mapping is missing attributes: {missing}")

        keys: list[str] = []
        self._group_specs: list[tuple[str, str, tuple[str, ...]]] = []
        for name, _, state_keys, fallback_key in self._SINGLE_GROUPS:
            keys.extend(state_keys)
            if fallback_key is not None:
                keys.append(fallback_key)
            self._group_specs.append((
                name, "single", tuple((*state_keys, fallback_key) if fallback_key else state_keys),
            ))
        for name, _, state_keys, fallback_key in self._MULTILABEL_GROUPS:
            keys.extend(state_keys)
            if fallback_key is not None:
                keys.append(fallback_key)
            self._group_specs.append((
                name, "multi", tuple((*state_keys, fallback_key) if fallback_key else state_keys),
            ))
        for column, positive_key, negative_key in self._BINARY_GROUPS:
            keys.extend((positive_key, negative_key))
            self._group_specs.append((
                column, "single", (positive_key, negative_key),
            ))

        if len(keys) != 50 or len(keys) != len(set(keys)):
            raise RuntimeError("Mixed category vocabulary must contain 50 unique states")
        if any(key not in CATEGORY_PROMPTS for key in keys):
            missing_prompts = sorted(set(keys) - set(CATEGORY_PROMPTS))
            raise RuntimeError(f"Missing mixed category prompts: {missing_prompts}")
        self.keys = keys
        self.prompts = [CATEGORY_PROMPTS[key] for key in keys]
        self._semantic_index = {key: index for index, key in enumerate(keys)}

    def category_specs(self) -> list[tuple[str, str, list[int]]]:
        """Return ``(name, kind, state_indices)`` for all 12 categories."""
        return [
            (name, kind, [self._semantic_index[key] for key in state_keys])
            for name, kind, state_keys in self._group_specs
        ]

    def category_indices(self) -> list[list[int]]:
        return [indices for _, _, indices in self.category_specs()]

    def multilabel_indices(self) -> list[list[int]]:
        return [indices for _, kind, indices in self.category_specs() if kind == "multi"]

    def single_indices(self) -> list[list[int]]:
        return [indices for _, kind, indices in self.category_specs() if kind == "single"]

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attribute_names):
            raise ValueError(f"Expected labels [B,40], got {tuple(labels.shape)}")
        binary = labels > 0.5
        semantic = torch.zeros(
            labels.shape[0], len(self.keys), dtype=torch.bool, device=labels.device,
        )
        for name, columns, state_keys, fallback_key in (
            *self._SINGLE_GROUPS, *self._MULTILABEL_GROUPS,
        ):
            columns_tensor = torch.tensor(
                [self._attribute_index[column] for column in columns], device=labels.device,
            )
            values = binary[:, columns_tensor]
            for offset, state_key in enumerate(state_keys):
                semantic[:, self._semantic_index[state_key]] = values[:, offset]
            if fallback_key is not None:
                semantic[:, self._semantic_index[fallback_key]] = ~values.any(dim=1)
        for column, positive_key, negative_key in self._BINARY_GROUPS:
            values = binary[:, self._attribute_index[column]]
            semantic[:, self._semantic_index[positive_key]] = values
            semantic[:, self._semantic_index[negative_key]] = ~values
        return semantic

    def project_40(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Project mixed state probabilities back to the official 40-bit order."""
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.keys):
            raise ValueError(f"Expected probabilities [B,50], got {tuple(probabilities.shape)}")
        positive_keys: dict[str, str] = {}
        for _, columns, state_keys, _ in (*self._SINGLE_GROUPS, *self._MULTILABEL_GROUPS):
            positive_keys.update(zip(columns, state_keys))
        for column, positive_key, _ in self._BINARY_GROUPS:
            positive_keys[column] = positive_key
        if set(positive_keys) != set(self.attribute_names):
            raise RuntimeError("Mixed category projection does not cover all 40 attributes")
        indices = [self._semantic_index[positive_keys[name]] for name in self.attribute_names]
        return probabilities[:, indices]
