"""Versioned configuration for general-purpose defect training projects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DefectClass:
    name: str
    description: str


@dataclass(frozen=True)
class LocalizationConfig:
    model: str
    prompts: tuple[str, ...]
    nms_iou: float = 0.70
    maximum_objects_per_image: int | None = None

    @property
    def prompt(self) -> str:
        """Backward-compatible access to the primary localization prompt."""

        return self.prompts[0]


@dataclass(frozen=True)
class CropConfig:
    scale: float = 1.25
    square: bool = False


@dataclass(frozen=True)
class ClassificationConfig:
    uncertain_label: str = "uncertain"
    automatic_acceptance: float = 0.90
    review_threshold: float = 0.70
    require_consensus: bool = True


@dataclass(frozen=True)
class SplitConfig:
    group_key: str
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class DefectProjectConfig:
    schema_version: int
    project_name: str
    localization: LocalizationConfig
    classes: tuple[DefectClass, ...]
    crop: CropConfig
    classification: ClassificationConfig
    split: SplitConfig

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)


def project_config_from_mapping(payload: Mapping[str, Any]) -> DefectProjectConfig:
    root = _mapping(payload, "configuration")
    version = root.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"Unsupported schema_version {version!r}; expected 1")
    project_name = _required_text(root.get("project_name"), "project_name")

    localization_raw = _mapping(root.get("localization"), "localization")
    nms_iou = _finite_number(localization_raw.get("nms_iou", 0.70), "localization.nms_iou")
    if not 0 < nms_iou <= 1:
        raise ValueError("localization.nms_iou must be in (0, 1]")
    maximum_objects = localization_raw.get("maximum_objects_per_image")
    if maximum_objects is not None:
        if isinstance(maximum_objects, bool) or not isinstance(maximum_objects, int) or maximum_objects <= 0:
            raise ValueError("localization.maximum_objects_per_image must be a positive integer")
    prompt_value = localization_raw.get("prompt")
    prompts_value = localization_raw.get("prompts")
    if prompt_value is not None and prompts_value is not None:
        raise ValueError("localization must define prompt or prompts, not both")
    if prompts_value is None:
        prompts = (_required_text(prompt_value, "localization.prompt"),)
    else:
        if not isinstance(prompts_value, list) or not prompts_value:
            raise ValueError("localization.prompts must be a non-empty list")
        prompts = tuple(
            _required_text(value, f"localization.prompts[{index}]")
            for index, value in enumerate(prompts_value)
        )
        if len(set(prompts)) != len(prompts):
            raise ValueError("localization prompts must be unique")
    localization = LocalizationConfig(
        model=_required_text(localization_raw.get("model"), "localization.model"),
        prompts=prompts,
        nms_iou=nms_iou,
        maximum_objects_per_image=maximum_objects,
    )

    classes_raw = root.get("classes")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError("classes must be a non-empty list")
    classes = tuple(
        DefectClass(
            name=_required_text(_mapping(item, f"classes[{index}]").get("name"), f"classes[{index}].name"),
            description=_required_text(
                _mapping(item, f"classes[{index}]").get("description"),
                f"classes[{index}].description",
            ),
        )
        for index, item in enumerate(classes_raw)
    )
    class_names = [item.name for item in classes]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique")

    crop_raw = _mapping(root.get("crop", {}), "crop")
    scale = _finite_number(crop_raw.get("scale", 1.25), "crop.scale")
    if scale < 1:
        raise ValueError("crop.scale must be at least 1.0")
    square = crop_raw.get("square", False)
    if not isinstance(square, bool):
        raise ValueError("crop.square must be a boolean")
    crop = CropConfig(scale=scale, square=square)

    classification_raw = _mapping(root.get("classification", {}), "classification")
    uncertain_label = _required_text(
        classification_raw.get("uncertain_label", "uncertain"), "classification.uncertain_label"
    )
    if uncertain_label in class_names:
        raise ValueError("classification.uncertain_label must not be a training class")
    automatic_acceptance = _finite_number(
        classification_raw.get("automatic_acceptance", 0.90),
        "classification.automatic_acceptance",
    )
    review_threshold = _finite_number(
        classification_raw.get("review_threshold", 0.70), "classification.review_threshold"
    )
    if not 0 <= review_threshold <= automatic_acceptance <= 1:
        raise ValueError("Classification thresholds must satisfy 0 <= review <= acceptance <= 1")
    require_consensus = classification_raw.get("require_consensus", True)
    if not isinstance(require_consensus, bool):
        raise ValueError("classification.require_consensus must be a boolean")
    classification = ClassificationConfig(
        uncertain_label=uncertain_label,
        automatic_acceptance=automatic_acceptance,
        review_threshold=review_threshold,
        require_consensus=require_consensus,
    )

    split_raw = _mapping(root.get("split"), "split")
    fractions = {
        name: _finite_number(split_raw.get(name), f"split.{name}")
        for name in ("train", "val", "test")
    }
    if any(value < 0 or value > 1 for value in fractions.values()):
        raise ValueError("Split fractions must be in [0, 1]")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.0")
    if fractions["train"] <= 0 or fractions["test"] <= 0:
        raise ValueError("Training and test split fractions must be positive")
    split = SplitConfig(
        group_key=_required_text(split_raw.get("group_key"), "split.group_key"), **fractions
    )
    return DefectProjectConfig(
        schema_version=1,
        project_name=project_name,
        localization=localization,
        classes=classes,
        crop=crop,
        classification=classification,
        split=split,
    )


def load_project_config(path: Path | str) -> DefectProjectConfig:
    """Load and fully validate a JSON or YAML project configuration."""

    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text())
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML configuration requires PyYAML; use a JSON configuration in minimal inference environments"
            ) from exc
        payload = yaml.safe_load(config_path.read_text())
    if payload is None:
        raise ValueError(f"Configuration is empty: {config_path}")
    return project_config_from_mapping(payload)
