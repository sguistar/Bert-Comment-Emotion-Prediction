from __future__ import annotations

import json
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# pythonw has no console streams. Some Transformers loading-report paths call
# isatty() unconditionally, so provide a quiet file-like stream in that mode.
_NULL_OUTPUT = None
if sys.stdout is None or sys.stderr is None:
    _NULL_OUTPUT = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _NULL_OUTPUT
    if sys.stderr is None:
        sys.stderr = _NULL_OUTPUT


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "outputs" / "sentiment" / "bert-v2" / "experiments"
BEST_ENSEMBLE_ID = "ensemble_diverse_weighted_calibrated_v2"
LABEL_ORDER = ["负向", "中性", "正向"]
DEPLOYMENT_EXPERIMENT_IDS = [
    BEST_ENSEMBLE_ID,
    "ensemble_embedding_rating5_baseline_calibrated_v1",
    "frozen_embedding_clspmean_bestencoder_seed42",
    "rating5mix_to_canonical3_ce_lr1e5_seed42",
    "rating5_to_canonical3_ce_seed42",
    "baseline_roberta_ce_seed2026",
    "rating5mix500_to_canonical3_ce_lr1e5_seed42",
    "cumbce_to_canonical3_ce_lr1e5_seed42",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid three-class probability vector: {values}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Probability vector has a non-positive sum")
    return values / total


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensemble_artifact_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Cheap cache key that changes whenever a deployed artifact is replaced."""

    files: list[Path] = []
    for experiment_id in DEPLOYMENT_EXPERIMENT_IDS:
        experiment_dir = EXPERIMENT_ROOT / experiment_id
        if not experiment_dir.is_dir():
            raise FileNotFoundError(experiment_dir)
        metrics_path = experiment_dir / "metrics.json"
        if metrics_path.is_file():
            files.append(metrics_path)
        classifier_path = experiment_dir / "classifier.pkl"
        if classifier_path.is_file():
            files.append(classifier_path)
        model_dir = experiment_dir / "best_model"
        if model_dir.is_dir():
            files.extend(path for path in model_dir.iterdir() if path.is_file())
    return tuple(
        (
            str(path.relative_to(ROOT)).replace("\\", "/"),
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(set(files))
    )


class TransformerProbabilityMember:
    def __init__(self, experiment_id: str, device: torch.device):
        self.experiment_id = experiment_id
        self.experiment_dir = EXPERIMENT_ROOT / experiment_id
        self.metrics = _read_json(self.experiment_dir / "metrics.json")
        self.model_dir = self.experiment_dir / "best_model"
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        self.max_length = int(self.metrics.get("config", {}).get("max_length", 384))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(device)
        if int(self.model.config.num_labels) != 3:
            raise ValueError(
                f"{experiment_id} has {self.model.config.num_labels} labels; expected 3"
            )
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def predict_proba(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        amp_enabled = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if amp_enabled else torch.float32,
            enabled=amp_enabled,
        ):
            logits = self.model(**encoded).logits
        return _normalize(F.softmax(logits.float(), dim=-1)[0].cpu().numpy())


class FrozenEmbeddingProbabilityMember:
    def __init__(self, experiment_id: str, device: torch.device):
        self.experiment_id = experiment_id
        self.experiment_dir = EXPERIMENT_ROOT / experiment_id
        self.metrics = _read_json(self.experiment_dir / "metrics.json")
        config = self.metrics.get("config", {})
        self.max_length = int(config.get("max_length", 384))
        self.pooling = str(config.get("pooling", "concat"))
        if self.pooling != "concat":
            raise ValueError(
                f"{experiment_id} uses unsupported pooling={self.pooling!r}"
            )
        encoder_dir = _resolve_project_path(str(config["model_name"]))
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_dir, local_files_only=True
        )
        sequence_model = AutoModelForSequenceClassification.from_pretrained(
            encoder_dir, local_files_only=True
        )
        self.encoder = sequence_model.base_model.to(device)
        del sequence_model
        self.encoder.eval()
        classifier_path = self.experiment_dir / "classifier.pkl"
        with classifier_path.open("rb") as stream:
            self.classifier = pickle.load(stream)
        self.classes = np.asarray(
            getattr(self.classifier, "classes_", []), dtype=np.int64
        )
        if sorted(self.classes.tolist()) != [0, 1, 2]:
            raise ValueError(
                f"{experiment_id} classifier classes are {self.classes.tolist()}; "
                "expected a permutation of [0, 1, 2]"
            )
        if not hasattr(self.classifier, "predict_proba"):
            raise TypeError(f"{experiment_id} classifier does not support predict_proba")
        self.device = device

    @torch.inference_mode()
    def predict_proba(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        amp_enabled = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if amp_enabled else torch.float32,
            enabled=amp_enabled,
        ):
            hidden = self.encoder(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = torch.cat([hidden[:, 0, :], mean_pool], dim=-1)
        features = F.normalize(pooled, p=2, dim=-1).cpu().numpy()
        raw_probabilities = self.classifier.predict_proba(features)[0]
        ordered_probabilities = np.zeros(3, dtype=np.float64)
        ordered_probabilities[self.classes] = raw_probabilities
        return _normalize(ordered_probabilities)


class BestSentimentEnsemble:
    """Deploy the highest-scoring comparable three-class validation ensemble."""

    def __init__(self, device: str | torch.device | None = None):
        if device is None:
            use_cuda = False
            if torch.cuda.is_available():
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                use_cuda = free_bytes >= 3 * 1024**3
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = torch.device(device)
        self._inference_lock = threading.RLock()
        self.experiment_dir = EXPERIMENT_ROOT / BEST_ENSEMBLE_ID
        self.metrics = _read_json(self.experiment_dir / "metrics.json")
        config = self.metrics["config"]
        self.sources = list(config["sources"])
        self.weights = np.asarray(config["weights"], dtype=np.float64)
        self.class_factors = np.asarray(config["class_factors"], dtype=np.float64)
        if len(self.sources) != 3 or self.weights.shape != (3,):
            raise ValueError("The deployed top-level ensemble must have exactly three sources")
        self.weights = self.weights / self.weights.sum()

        nested_id = self.sources[0]
        nested_metrics = _read_json(EXPERIMENT_ROOT / nested_id / "metrics.json")
        nested_config = nested_metrics["config"]
        self.nested_members = list(nested_config["members"])
        self.nested_class_factors = np.asarray(
            nested_config["class_factors"], dtype=np.float64
        )
        if len(self.nested_members) != 3:
            raise ValueError("The deployed nested ensemble must have exactly three members")

        self.members: dict[str, Any] = {}
        for experiment_id in self.nested_members:
            experiment_dir = EXPERIMENT_ROOT / experiment_id
            if (experiment_dir / "classifier.pkl").is_file():
                member = FrozenEmbeddingProbabilityMember(experiment_id, self.device)
            else:
                member = TransformerProbabilityMember(experiment_id, self.device)
            self.members[experiment_id] = member
        for experiment_id in self.sources[1:]:
            self.members[experiment_id] = TransformerProbabilityMember(
                experiment_id, self.device
            )

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = str(text).strip()
        if not normalized_text:
            raise ValueError("评论文本不能为空")

        with self._inference_lock:
            return self._predict_unlocked(normalized_text)

    def _predict_unlocked(self, normalized_text: str) -> dict[str, Any]:

        leaf_probabilities = {
            experiment_id: member.predict_proba(normalized_text)
            for experiment_id, member in self.members.items()
        }
        nested = np.mean(
            [leaf_probabilities[experiment_id] for experiment_id in self.nested_members],
            axis=0,
        )
        nested = _normalize(nested * self.nested_class_factors)
        top_level = (
            self.weights[0] * nested
            + self.weights[1] * leaf_probabilities[self.sources[1]]
            + self.weights[2] * leaf_probabilities[self.sources[2]]
        )
        probabilities = _normalize(top_level * self.class_factors)
        label_id = int(probabilities.argmax())
        return {
            "label": LABEL_ORDER[label_id],
            "confidence": float(probabilities[label_id]),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(LABEL_ORDER)
            },
            "model_id": BEST_ENSEMBLE_ID,
            "device": str(self.device),
        }


def load_best_validation_metrics() -> dict[str, Any]:
    payload = _read_json(EXPERIMENT_ROOT / BEST_ENSEMBLE_ID / "metrics.json")
    return dict(payload.get("valid_metrics", {}))
