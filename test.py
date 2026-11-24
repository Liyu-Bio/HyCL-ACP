"""Evaluation and inference pipeline for HyCL-ACP."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model import (
    AAC_DIMENSIONS,
    CrossModalNeuralPathway,
    DPC_DIMENSIONS,
    ESM_LAYER,
    ESM_MODEL,
    ESM_POOLING,
    SVM_C,
    build_PSRA_svm,
    cached_PSRA_esm_view,
    make_data,
    make_seq_data,
    PSRA_aac_dpc_features,
    sha256,
)

ROOT = Path(__file__).resolve().parent


def fit_PSRA_binary_svm(features: np.ndarray, labels: np.ndarray,
                        indices: np.ndarray, c_value: float = SVM_C):
    """Fit the PSRA classifier on the requested partition."""
    classifier = build_PSRA_svm(c_value)
    classifier.fit(features[indices], labels[indices])
    return classifier


def PSRA_multilabel_probabilities(
        features: np.ndarray, labels: np.ndarray,
        train_indices: np.ndarray, eval_indices: np.ndarray,
        c_value: float = SVM_C):
    """Fit one PSRA classifier per label and return positive probabilities."""
    columns, classifiers = [], []
    for label_index in range(labels.shape[1]):
        classifier = fit_PSRA_binary_svm(
            features, labels[:, label_index], train_indices, c_value)
        positive_column = list(classifier[-1].classes_).index(1)
        columns.append(
            classifier.predict_proba(features[eval_indices])[:, positive_column])
        classifiers.append(classifier)
    return np.stack(columns, axis=1), classifiers


BRANCH_NAMES = ("PSRA", "CIF")


BINARY_FUSION_PRIOR = np.asarray([0.80, 0.20], dtype=np.float64)


MULTILABEL_FUSION_PRIOR = np.asarray([0.75, 0.25], dtype=np.float64)


FUSION_PRIOR = BINARY_FUSION_PRIOR.copy()


FUSION_PRIOR_STRENGTH = 1.0


FUSION_STEP = 0.01


MIN_BRANCH_WEIGHT = 0.05


def simplex_weights(n_branches: int, step: float = 0.1,
                    min_weight: float = MIN_BRANCH_WEIGHT) -> list[np.ndarray]:
    """Enumerate deterministic convex weights while retaining every branch."""
    denominator = int(round(1.0 / step))
    if not np.isclose(denominator * step, 1.0):
        raise ValueError("step must divide one exactly")
    weights = []
    for cuts in itertools.combinations_with_replacement(range(denominator + 1),
                                                         n_branches - 1):
        boundaries = (0, *cuts, denominator)
        counts = np.diff(boundaries)
        current = counts.astype(np.float64) / denominator
        if np.all(current >= min_weight - 1e-12):
            weights.append(current)
    if not weights:
        raise ValueError("step/min_weight leave no feasible convex weights")
    return weights


def binary_cross_entropy(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Mean BCE used by the common automatic fusion learner."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(probability)
                          + (1.0 - y_true) * np.log(1.0 - probability)))


def learn_convex_fusion(y_valid: np.ndarray, branch_probabilities: np.ndarray,
                        step: float = 0.1,
                        min_weight: float = MIN_BRANCH_WEIGHT,
                        prior_strength: float = FUSION_PRIOR_STRENGTH,
                        prior: np.ndarray = FUSION_PRIOR) -> dict[str, object]:
    """Learn prior-regularized convex fusion on validation data.

    ``branch_probabilities`` has shape ``[branches, rows]`` for binary data or
    ``[branches, rows, labels]`` for multilabel data.  Test labels never enter
    this function.  The fixed prior prevents a small validation set from moving
    the weights without enough BCE improvement.
    """
    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if probabilities.shape[0] != len(BRANCH_NAMES):
        raise ValueError(f"Expected {len(BRANCH_NAMES)} branches, got "
                         f"{probabilities.shape[0]}")
    if probabilities.shape[1:] != np.asarray(y_valid).shape:
        raise ValueError("Validation labels and branch probabilities do not align")
    if prior.shape != (probabilities.shape[0],) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("Fusion prior must contain one normalized weight per branch")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    best = None
    for weights in simplex_weights(probabilities.shape[0], step, min_weight):
        fused = np.tensordot(weights, probabilities, axes=(0, 0))
        loss = binary_cross_entropy(y_valid, fused)
        distance = float(np.sum((weights - prior) ** 2))
        prior_penalty = prior_strength * distance
        regularized_objective = loss + prior_penalty
        key = (regularized_objective, loss, distance, tuple(weights.tolist()))
        if best is None or key < best[0]:
            best = (key, weights, loss, distance, prior_penalty)
    return {
        "weights": best[1],
        "validation_bce": best[2],
        "prior_squared_distance": best[3],
        "prior_penalty": best[4],
        "regularized_objective": best[0][0],
        "prior": prior,
        "prior_strength": prior_strength,
        "step": step,
        "min_weight": min_weight,
        "objective": "validation BCE + prior_strength * squared L2 distance to prior",
    }


def validation_classification_metrics(y_true: np.ndarray,
                                      probability: np.ndarray) -> dict[str, float]:
    """Metrics shared by binary and multilabel validation selection."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    if y_true.shape != probability.shape:
        raise ValueError("Validation labels and probabilities do not align")
    average = "macro" if y_true.ndim == 2 else None
    return {
        "accuracy": float(np.mean((probability >= 0.5) == y_true)),
        "auroc": float(roc_auc_score(y_true, probability, average=average)),
    }


def learn_prior_regularized_fusion_candidates(
        y_valid: np.ndarray,
        candidates: dict[str, np.ndarray],
        step: float = 0.05,
        min_weight: float = MIN_BRANCH_WEIGHT,
        prior_strength: float = FUSION_PRIOR_STRENGTH,
        prior: np.ndarray = FUSION_PRIOR) -> dict[str, object]:
    """Select a graph candidate and prior-regularized weights on validation data."""
    best = None
    for candidate_id, branch_probabilities in sorted(candidates.items()):
        selection = learn_convex_fusion(
            y_valid, branch_probabilities, step=step,
            min_weight=min_weight, prior_strength=prior_strength, prior=prior)
        weights = np.asarray(selection["weights"], dtype=np.float64)
        fused = PPI_fuse_probabilities(weights, branch_probabilities)
        metric = validation_classification_metrics(y_valid, fused)
        key = (selection["regularized_objective"],
               selection["validation_bce"], candidate_id)
        if best is None or key < best[0]:
            best = (key, candidate_id, selection, metric)
    selected = best[2]
    return {
        "cross_attention_candidate": best[1],
        "weights": selected["weights"],
        "threshold": 0.5,
        "validation_accuracy": best[3]["accuracy"],
        "validation_auroc": best[3]["auroc"],
        "validation_bce": selected["validation_bce"],
        "prior_squared_distance": selected["prior_squared_distance"],
        "prior_penalty": selected["prior_penalty"],
        "regularized_objective": selected["regularized_objective"],
        "prior": selected["prior"],
        "prior_strength": selected["prior_strength"],
        "step": step,
        "minimum_branch_weight": min_weight,
        "selection_rule": selected["objective"],
    }


def learn_PPI_weights(
        y_valid: np.ndarray,
        candidates: dict[str, np.ndarray],
        step: float = FUSION_STEP,
        min_weight: float = MIN_BRANCH_WEIGHT,
        prior_strength: float = FUSION_PRIOR_STRENGTH,
        prior: np.ndarray = FUSION_PRIOR) -> dict[str, object]:
    """Select fusion weights by validation error with a structural prior.

    The primary term is the validation classification error at the deployment
    threshold of 0.5, so the selector is aligned with the reported ACC.  The
    squared distance to the provided task-specific prior discourages a small
    validation split from moving the two-branch architecture unnecessarily.
    BCE is used only as a deterministic tie-break after the stated objective.
    """
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (len(BRANCH_NAMES),) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("Fusion prior must contain one normalized weight per branch")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")

    best = None
    for candidate_id, branch_probabilities in sorted(candidates.items()):
        probabilities = np.asarray(branch_probabilities, dtype=np.float64)
        if probabilities.shape[0] != len(BRANCH_NAMES):
            raise ValueError(
                f"Expected {len(BRANCH_NAMES)} branches, got {probabilities.shape[0]}")
        if probabilities.shape[1:] != np.asarray(y_valid).shape:
            raise ValueError("Validation labels and branch probabilities do not align")
        for weights in simplex_weights(probabilities.shape[0], step, min_weight):
            fused = PPI_fuse_probabilities(weights, probabilities)
            metric = validation_classification_metrics(y_valid, fused)
            validation_error = 1.0 - metric["accuracy"]
            loss = binary_cross_entropy(y_valid, fused)
            distance = float(np.sum((weights - prior) ** 2))
            prior_penalty = prior_strength * distance
            objective = validation_error + prior_penalty
            key = (objective, loss, distance, candidate_id,
                   tuple(weights.tolist()))
            if best is None or key < best[0]:
                best = (key, candidate_id, weights, metric, loss,
                        validation_error, distance, prior_penalty)

    return {
        "cross_attention_candidate": best[1],
        "weights": best[2],
        "threshold": 0.5,
        "validation_accuracy": best[3]["accuracy"],
        "validation_auroc": best[3]["auroc"],
        "validation_bce": best[4],
        "validation_error": best[5],
        "prior_squared_distance": best[6],
        "prior_penalty": best[7],
        "regularized_objective": best[0][0],
        "prior": prior,
        "prior_strength": prior_strength,
        "step": step,
        "minimum_branch_weight": min_weight,
        "selection_rule": (
            "1 - validation accuracy + prior_strength * squared L2 distance "
            "to prior; validation BCE is the tie-break"
        ),
    }


def PPI_fuse_probabilities(weights: np.ndarray,
                            branch_probabilities: np.ndarray) -> np.ndarray:
    """Apply Prediction Integration and Prediction (PPI)."""
    return np.tensordot(np.asarray(weights, dtype=np.float64),
                        np.asarray(branch_probabilities, dtype=np.float64),
                        axes=(0, 0))


def binary_metrics(y_true: np.ndarray, probability: np.ndarray,
                   threshold: float) -> dict[str, float | int]:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def multilabel_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(probability) >= 0.5
    y_true = np.asarray(y_true)
    matches = prediction == y_true
    macro_accuracy = float(np.mean(np.mean(matches, axis=0)))
    micro_accuracy = float(np.mean(matches.reshape(-1)))
    return {
        "label_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "subset_accuracy": float(np.mean(np.all(prediction == y_true, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, prediction)),
        "macro_precision": float(precision_score(
            y_true, prediction, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(
            y_true, prediction, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, prediction, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(
            y_true, prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, prediction, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(
            y_true, prediction, average="micro", zero_division=0)),
        "macro_mcc": float(np.mean([
            matthews_corrcoef(y_true[:, index], prediction[:, index])
            for index in range(y_true.shape[1])
        ])),
        "micro_mcc": float(matthews_corrcoef(
            y_true.reshape(-1), prediction.reshape(-1))),
        "macro_auroc": float(roc_auc_score(y_true, probability, average="macro")),
        "micro_auroc": float(roc_auc_score(y_true, probability, average="micro")),
        "macro_auprc": float(average_precision_score(y_true, probability, average="macro")),
        "micro_auprc": float(average_precision_score(y_true, probability, average="micro")),
    }


def resolve_data(path: Path) -> tuple[str, Path]:
    """Return (task, canonical data path) from file layout and columns."""
    path = path.expanduser().resolve()
    if path.is_dir():
        train, test = path / "train.csv", path / "test.csv"
        if not train.exists() or not test.exists():
            raise ValueError(f"Binary directory must contain train.csv and test.csv: {path}")
        for current in (train, test):
            columns = set(pd.read_csv(current, nrows=1).columns)
            if not {"Feature", "Label"} <= columns:
                raise ValueError(f"Expected Feature/Label columns in {current}")
        return "binary", path
    if path.is_file():
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        columns = set(pd.read_csv(path, sep=separator, nrows=1).columns)
        if {"text", "label"} <= columns:
            return "multilabel", path
        raise ValueError(f"Expected text/label columns in {path}")
    raise FileNotFoundError(path)


def dataset_key(data: Path) -> str:
    """Create a readable collision-resistant runtime ID from the canonical path."""
    task, data = resolve_data(data)
    label = data.name if task == "binary" else data.stem
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or task
    digest = hashlib.sha256(str(data).encode()).hexdigest()[:10]
    return f"{label}_{digest}"


def binary_frames(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task, data = resolve_data(data)
    if task != "binary":
        raise ValueError(f"Not a binary dataset directory: {data}")
    return pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")


def multilabel_frame(data: Path) -> pd.DataFrame:
    task, data = resolve_data(data)
    if task != "multilabel":
        raise ValueError(f"Not a multilabel dataset file: {data}")
    separator = "\t" if data.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(data, sep=separator, dtype=str, keep_default_na=False)


DATASETS = {
    "ACP": {
        "task": "binary",
        "data": ROOT / "data/binary/ACP",
    },
    "CancerPPD": {
        "task": "multilabel",
        "data": ROOT / "data/multilabel/CancerPPD.tsv",
    },
}


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def resolve_dataset_name(name: str) -> str:
    """Resolve a dataset name case-insensitively and ignoring punctuation."""
    lookup = {_normalized(canonical): canonical for canonical in DATASETS}
    key = _normalized(name)
    if key not in lookup:
        available = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")
    return lookup[key]


def dataset_spec(name: str) -> tuple[str, dict[str, object]]:
    canonical = resolve_dataset_name(name)
    return canonical, DATASETS[canonical]


def require_local_file(path: Path, description: str) -> None:
    """Require a regular project-local artifact without following legacy links."""
    if path.is_symlink():
        raise RuntimeError(
            f"{description} must be stored in this project, not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing local {description}: {path}")


def prepare_dataset(name: str) -> tuple[str, dict[str, object], Path, Path | None]:
    """Resolve a name and validate project-local ESM and graph inputs."""
    canonical, registered_spec = dataset_spec(name)
    spec = dict(registered_spec)
    data = Path(spec["data"]).resolve()
    key = dataset_key(data)
    cache_root = ROOT / "cache" / "esm" / ESM_MODEL
    if spec["task"] == "binary":
        for split in ("train", "test"):
            require_local_file(
                cache_root / f"{key}_{split}.npz",
                f"ESM {split} cache (run python train.py ACP first)",
            )
        require_local_file(
            ROOT / "artifacts" / "graph" / f"{key}_seed5.npz",
            "graph probability artifact (run extract_graph_probabilities.py first)",
        )
        return canonical, spec, data, None

    cache_path = cache_root / f"{key}.npz"
    source_run = (
        ROOT / "results" / "multilabel" /
        f"cross_attention_{key}_seed5"
    ).resolve()
    if not cache_path.is_file() or len(list(source_run.glob("fold_*/graph.pt"))) != 10:
        # dataset_key historically included the absolute project path.  When a
        # complete project is moved, discover the one locally complete legacy
        # artifact set; multilabel_pipeline still verifies sequence order.
        label = key.rsplit("_", 1)[0]
        candidates = []
        for candidate_cache in sorted(cache_root.glob(f"{label}_*.npz")):
            candidate_key = candidate_cache.stem
            candidate_run = (
                ROOT / "results" / "multilabel" /
                f"cross_attention_{candidate_key}_seed5"
            ).resolve()
            if len(list(candidate_run.glob("fold_*/graph.pt"))) == 10:
                candidates.append((candidate_key, candidate_cache, candidate_run))
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not resolve one complete local multilabel artifact set; "
                f"expected key={key}, candidates={[item[0] for item in candidates]}")
        key, cache_path, source_run = candidates[0]
    require_local_file(
        cache_path, "ESM cache (run python train.py CancerPPD first)")
    spec["runtime_key"] = key
    if len(list(source_run.glob("fold_*/graph.pt"))) != 10:
        raise RuntimeError(f"Expected ten cross-attention checkpoints in {source_run}")
    return canonical, spec, data, source_run


SUMMARY_COLUMNS = (
    "dataset_id", "source_data", "accuracy", "auroc", "auprc", "f1",
    "precision", "sensitivity", "specificity", "mcc", "tn", "fp", "fn",
    "tp", "svm_C", "cross_attention_candidate", "weights", "threshold",
)


def cross_attention_files(key: str, valid_indices: np.ndarray,
                          n_test: int) -> list[Path]:
    """Find only the seed-5 artifact, with a legacy-name fallback."""
    artifact_dir = ROOT / "artifacts" / "graph"
    exact = artifact_dir / f"{key}_seed5.npz"
    if exact.exists():
        return [exact]
    compatible = []
    for path in sorted(artifact_dir.glob("*.npz")):
        # Historical seed-5 files had no seed suffix.  Never admit seed-1
        # artifacts into the formal selector.
        if "seed" in path.stem.lower() and not path.stem.lower().endswith("seed5"):
            continue
        with np.load(path) as saved:
            if ("valid_indices" in saved and "test_probability" in saved
                    and np.array_equal(saved["valid_indices"], valid_indices)
                    and len(saved["test_probability"]) == n_test):
                compatible.append(path)
    if not compatible:
        raise FileNotFoundError(f"No seed-5 cross-attention artifact for runtime ID {key}")
    if len(compatible) > 1:
        raise RuntimeError(
            f"Ambiguous legacy seed-5 artifacts for runtime ID {key}: {compatible}")
    return compatible


def run_binary_dataset(data: Path) -> dict:
    task, data = resolve_data(data)
    if task != "binary":
        raise ValueError(f"Expected a binary data directory, got {task}")
    key = dataset_key(data)
    train, test = binary_frames(data)
    train_path, test_path = data / "train.csv", data / "test.csv"
    train_sequences = train["Feature"].astype(str).str.strip().str.upper().tolist()
    test_sequences = test["Feature"].astype(str).str.strip().str.upper().tolist()
    y_train = train["Label"].to_numpy(dtype=int)
    y_test = test["Label"].to_numpy(dtype=int)
    train_indices, valid_indices = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=5, stratify=y_train)

    cache_dir = ROOT / "cache" / "esm" / ESM_MODEL
    train_cache = np.load(cache_dir / f"{key}_train.npz")
    test_cache = np.load(cache_dir / f"{key}_test.npz")
    if (train_cache["sequences"].tolist() != train_sequences
            or test_cache["sequences"].tolist() != test_sequences):
        raise RuntimeError("ESM cache sequence order changed")
    feature_name = f"layer{ESM_LAYER}_{ESM_POOLING}"
    x_train = np.concatenate([
        cached_PSRA_esm_view(train_cache, feature_name),
        PSRA_aac_dpc_features(train_sequences),
    ], axis=1)
    x_test = np.concatenate([
        cached_PSRA_esm_view(test_cache, feature_name),
        PSRA_aac_dpc_features(test_sequences),
    ], axis=1)
    graph_files = cross_attention_files(key, valid_indices, len(test))
    cross_valid, cross_test, checkpoints = {}, {}, {}
    for path in graph_files:
        graph_id = path.stem.removeprefix(f"{key}_")
        saved = np.load(path)
        if not np.array_equal(saved["valid_indices"], valid_indices):
            raise RuntimeError("Fixed validation indices changed")
        cross_valid[graph_id] = saved["valid_probability"]
        cross_test[graph_id] = saved["test_probability"]
        checkpoints[graph_id] = str(saved["checkpoint"].item())

    if len(cross_test) != 1:
        raise RuntimeError(f"Expected one seed-5 cross-attention candidate, got {cross_test}")
    validation_model = fit_PSRA_binary_svm(x_train, y_train, train_indices, SVM_C)
    esm_valid = validation_model.predict_proba(x_train[valid_indices])[:, 1]
    candidates = {
        graph_id: np.stack([esm_valid, cross_valid[graph_id]])
        for graph_id in cross_test
    }
    selection = learn_PPI_weights(
        y_train[valid_indices], candidates, step=FUSION_STEP,
        prior=np.asarray(BINARY_FUSION_PRIOR, dtype=float))
    graph_id = selection["cross_attention_candidate"]
    weights = np.asarray(selection["weights"], dtype=float)
    selection["weights"] = dict(zip(BRANCH_NAMES, weights.tolist()))
    selection["prior"] = dict(zip(
        BRANCH_NAMES, np.asarray(selection["prior"], dtype=float).tolist()))
    selection["svm_C"] = SVM_C
    selection["svm_C_selection"] = "fixed globally"
    c_value = SVM_C
    final_model = fit_PSRA_binary_svm(x_train, y_train, np.arange(len(train)), c_value)
    esm_test = final_model.predict_proba(x_test)[:, 1]
    test_branches = np.stack([esm_test, cross_test[graph_id]])
    probability = PPI_fuse_probabilities(weights, test_branches)
    metric = binary_metrics(y_test, probability, selection["threshold"])

    output_dir = ROOT / "results" / "two_branch" / "binary" / key
    model_dir = ROOT / "models" / "two_branch" / "binary" / key
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_split.json").write_text(json.dumps({
        "method": "train_test_split(test_size=0.2, random_state=5, stratify=y)",
        "train": train_indices.tolist(), "valid": valid_indices.tolist()}, indent=2))
    joblib.dump({"branch": BRANCH_NAMES[0], "model": final_model},
                model_dir / "esm_predictor.joblib", compress=3)
    predictions = pd.DataFrame({
        "row": np.arange(len(test)), "sequence": test_sequences, "label": y_test,
        "PSKA_probability": esm_test,
        "bidirectional_cross_attention_probability": cross_test[graph_id],
        "probability": probability,
        "prediction": (probability >= selection["threshold"]).astype(int)})
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    prior_description = "/".join(
        f"{weight:.2f}" for weight in BINARY_FUSION_PRIOR)
    payload = {"dataset_id": key, "source_data": str(data),
               "top_level_branches": list(BRANCH_NAMES),
               "protocol": f"validation-ACC-selected prior-regularized fusion; objective=1-ACC+L2-to-{prior_description}-prior, BCE tie-break; 0.01 grid, strength=1.0; stratified 80/20 validation split, random_state=5",
               "esm_branch": {"model": ESM_MODEL, "layer": ESM_LAYER,
                              "paper_name": "PSKA",
                              "pooling": ESM_POOLING,
                              "input_features": "frozen ESM embedding + AAC + DPC",
                              "auxiliary_features": "AAC20+DPC400",
                              "aac_dimensions": AAC_DIMENSIONS,
                              "dpc_dimensions": DPC_DIMENSIONS,
                              "total_dimensions": int(x_train.shape[1]),
                              "classifier": "balanced RBF-SVM",
                              "C": SVM_C, "C_selection": "fixed globally"},
               "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
               "selection": selection,
               "cross_attention_checkpoints": checkpoints,
               "metrics": metric}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{key} | final test metrics (seed=5)")
    print(
        "ACC={accuracy:.6f} | AUC={auroc:.6f} | AUPRC={auprc:.6f} | "
        "F1={f1:.6f} | PRE={precision:.6f} | SE={sensitivity:.6f} | "
        "SP={specificity:.6f} | MCC={mcc:.6f}".format(**metric)
    )
    print("TN={tn} | FP={fp} | FN={fn} | TP={tp}".format(**metric))
    print(
        f"C={selection['svm_C']} | threshold={selection['threshold']:.3f} | "
        f"weights={selection['weights']} | "
        f"cross_attention={selection['cross_attention_candidate']}"
    )
    return payload


def update_binary_summary(result: dict) -> None:
    """Insert or replace one dataset row in the binary summary."""
    output = ROOT / "results" / "two_branch"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "binary_summary.csv"
    current = pd.DataFrame([{
        "dataset_id": result["dataset_id"], "source_data": result["source_data"],
        **result["metrics"],
        "svm_C": result["selection"]["svm_C"],
        "cross_attention_candidate": result["selection"]["cross_attention_candidate"],
        "weights": json.dumps(result["selection"]["weights"]),
        "threshold": result["selection"]["threshold"]}])
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        id_column = "dataset_id" if "dataset_id" in previous else None
        if id_column:
            previous = previous[~previous[id_column].isin(current["dataset_id"])]
            current = pd.concat([previous, current])
    current = current.reindex(columns=SUMMARY_COLUMNS)
    current.sort_values("dataset_id").to_csv(summary_path, index=False)


def read_labels(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    labels, rows = [], []
    for value in frame["label"]:
        current = {item.strip() for item in value.split(",")}
        rows.append(current)
        for item in [item.strip() for item in value.split(",")]:
            if item not in labels:
                labels.append(item)
    return np.asarray([[label in row for label in labels] for row in rows], dtype=int), labels


def validate_splits(partitions: list[dict], n_samples: int) -> None:
    if len(partitions) != 10:
        raise ValueError(f"Expected 10 folds, found {len(partitions)}")
    outer_rows = []
    for expected_fold, split in enumerate(partitions, 1):
        if split["fold"] != expected_fold:
            raise ValueError("Fold numbering changed")
        train, valid, test = map(set, (split["train"], split["valid"], split["test"]))
        if train & valid or train & test or valid & test:
            raise ValueError(f"Fold {expected_fold} has overlapping partitions")
        outer_rows.extend(test)
    if sorted(outer_rows) != list(range(n_samples)):
        raise ValueError("Outer test folds are not an exact partition of all rows")


def cross_attention_probabilities(checkpoint: Path, sequences_tensor: torch.Tensor,
                                  graphs: list, indices: np.ndarray,
                                  batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CrossModalNeuralPathway(SimpleNamespace(**payload["architecture"]))
    model = model.to(device).eval()
    model.load_state_dict(payload["state_dict"])
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start:start + batch_size]
            result = model(sequences_tensor[current].to(device),
                           dgl.batch([graphs[index] for index in current]).to(device))
            output.append(result["out"].sigmoid().cpu().numpy())
    return np.concatenate(output)


def run_multilabel_dataset(data: Path, source_run: Path,
                runtime_key: str | None = None) -> dict:
    task, data = resolve_data(data)
    if task != "multilabel":
        raise ValueError(f"Expected a multilabel file, got {task}")
    key = runtime_key or dataset_key(data)
    frame = multilabel_frame(data)
    sequences = frame["text"].astype(str).str.strip().str.upper().tolist()
    y, labels = read_labels(frame)
    source_run = source_run.resolve()
    split_path = source_run / "splits.json"
    partitions = json.loads(split_path.read_text())
    validate_splits(partitions, len(frame))
    cache = np.load(ROOT / "cache" / "esm" / ESM_MODEL / f"{key}.npz")
    if cache["sequences"].tolist() != sequences:
        raise RuntimeError("ESM cache sequence order changed")
    feature_name = f"layer{ESM_LAYER}_{ESM_POOLING}"
    esm_features = np.concatenate([
        cached_PSRA_esm_view(cache, feature_name),
        PSRA_aac_dpc_features(sequences),
    ], axis=1)
    graphs = make_data(sequences)
    sequence_tensor = make_seq_data(sequences, max(map(len, sequences)))

    oof = np.full(y.shape, np.nan, dtype=float)
    fold_ids = np.zeros(len(frame), dtype=int)
    fold_reports, fusion_reports = [], []
    model_root = ROOT / "models" / "two_branch" / "multilabel" / key
    model_root.mkdir(parents=True, exist_ok=True)
    for split in partitions:
        fold = split["fold"]
        train = np.asarray(split["train"], dtype=int)
        valid = np.asarray(split["valid"], dtype=int)
        development = np.r_[train, valid]
        test = np.asarray(split["test"], dtype=int)
        checkpoint = source_run / f"fold_{fold:02d}" / "graph.pt"
        graph_saved = np.load(source_run / f"fold_{fold:02d}" / "graph_predictions.npz")
        if not np.array_equal(graph_saved["rows"], test):
            raise RuntimeError(f"Graph outer-test rows changed in fold {fold}")
        cross_attention_test = graph_saved["probabilities"]
        cross_attention_valid = cross_attention_probabilities(
            checkpoint, sequence_tensor, graphs, valid)
        esm_valid, _ = PSRA_multilabel_probabilities(
            esm_features, y, train, valid, SVM_C)
        esm_test, classifiers = PSRA_multilabel_probabilities(
            esm_features, y, development, test, SVM_C)
        selection = learn_PPI_weights(
            y[valid], {f"fold_{fold:02d}": np.stack([
                esm_valid, cross_attention_valid])},
            step=FUSION_STEP,
            prior=np.asarray(MULTILABEL_FUSION_PRIOR, dtype=float))
        weights = np.asarray(selection["weights"], dtype=float)
        probability = PPI_fuse_probabilities(
            weights, np.stack([esm_test, cross_attention_test]))
        oof[test] = probability
        fold_ids[test] = fold
        fold_reports.append({"fold": fold, **multilabel_metrics(y[test], probability)})
        fusion_reports.append({
            "fold": fold,
            "weights": dict(zip(BRANCH_NAMES, weights.tolist())),
            "prior": dict(zip(
                BRANCH_NAMES,
                np.asarray(selection["prior"], dtype=float).tolist())),
            "validation_accuracy": selection["validation_accuracy"],
            "validation_auroc": selection["validation_auroc"],
            "validation_bce": selection["validation_bce"],
            "validation_error": selection["validation_error"],
            "prior_squared_distance": selection["prior_squared_distance"],
            "prior_penalty": selection["prior_penalty"],
            "regularized_objective": selection["regularized_objective"],
            "prior_strength": selection["prior_strength"],
            "step": selection["step"],
            "minimum_branch_weight": selection["minimum_branch_weight"],
            "svm_C": SVM_C,
            "svm_C_selection": "fixed globally",
            "selection_rule": selection["selection_rule"],
        })
        joblib.dump({"labels": labels, "top_level_branches": BRANCH_NAMES,
                     "esm_models": classifiers, "fusion_weights": weights,
                     "fusion_prior": np.asarray(MULTILABEL_FUSION_PRIOR),
                     "fusion_policy": selection["selection_rule"]},
                    model_root / f"fold_{fold:02d}.joblib", compress=3)
        print(f"fold {fold:02d}: weights={weights.tolist()}", flush=True)
    if not np.isfinite(oof).all() or not np.all(fold_ids > 0):
        raise RuntimeError("OOF predictions are incomplete")

    metric = multilabel_metrics(y, oof)
    output_dir = ROOT / "results" / "two_branch" / "multilabel" / key
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits.json").write_text(
        json.dumps(partitions, ensure_ascii=False, indent=2), encoding="utf-8")
    output = frame.copy()
    output["fold"] = fold_ids
    for label_index, label in enumerate(labels):
        output[f"{label}_true"] = y[:, label_index]
        output[f"{label}_probability"] = oof[:, label_index]
    output.to_csv(output_dir / "oof_predictions.csv", index=False)
    prior_description = "/".join(
        f"{weight:.2f}" for weight in MULTILABEL_FUSION_PRIOR)
    payload = {"dataset_id": key, "source_data": str(data),
               "source_cross_attention_run": str(source_run),
               "top_level_branches": list(BRANCH_NAMES),
               "protocol": f"strict 10-fold outer OOF; per-fold validation-ACC-selected prior-regularized fusion; objective=1-ACC+L2-to-{prior_description}-prior, BCE tie-break; 0.01 grid, strength=1.0",
               "n_samples": len(frame), "labels": labels,
               "data_sha256": sha256(data), "splits_sha256": sha256(split_path),
               "esm_branch": {"model": f"{ESM_MODEL} (frozen)",
                              "paper_name": "PSKA",
                              "layer": ESM_LAYER, "pooling": ESM_POOLING,
                              "input_features": "frozen ESM embedding + AAC + DPC",
                              "auxiliary_features": "AAC20+DPC400",
                              "aac_dimensions": AAC_DIMENSIONS,
                              "dpc_dimensions": DPC_DIMENSIONS,
                              "total_dimensions": int(esm_features.shape[1]),
                              "classifier": "balanced RBF-SVM",
                              "C": SVM_C, "C_selection": "fixed globally"},
               "fusion": {"method": "validation label error + squared-L2 prior regularization; BCE tie-break",
                          "prior": dict(zip(
                              BRANCH_NAMES,
                              np.asarray(MULTILABEL_FUSION_PRIOR).tolist())),
                          "folds": fusion_reports},
               "metrics": metric, "fold_metrics": fold_reports}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{key} | strict 10-fold OOF metrics (seed=5, C={SVM_C})")
    print(
        "Label-ACC={label_accuracy:.6f} | Macro-ACC={macro_accuracy:.6f} | "
        "Micro-ACC={micro_accuracy:.6f} | Subset-ACC={subset_accuracy:.6f} | "
        "Hamming-Loss={hamming_loss:.6f}".format(**metric)
    )
    print(
        "Macro: AUC={macro_auroc:.6f} | AUPRC={macro_auprc:.6f} | "
        "F1={macro_f1:.6f} | PRE={macro_precision:.6f} | "
        "REC={macro_recall:.6f} | MCC={macro_mcc:.6f}".format(**metric)
    )
    print(
        "Micro: AUC={micro_auroc:.6f} | AUPRC={micro_auprc:.6f} | "
        "F1={micro_f1:.6f} | PRE={micro_precision:.6f} | "
        "REC={micro_recall:.6f} | MCC={micro_mcc:.6f}".format(**metric)
    )
    return payload


def update_multilabel_summary(result: dict) -> None:
    """Insert or replace one dataset row in the multilabel summary."""
    output = ROOT / "results" / "two_branch"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "multilabel_summary.csv"
    current = pd.DataFrame([{
        "dataset_id": result["dataset_id"], "source_data": result["source_data"],
        **result["metrics"]}])
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        if "dataset_id" in previous:
            previous = previous[~previous["dataset_id"].isin(current["dataset_id"])]
            current = pd.concat([previous, current])
    current.sort_values("dataset_id").to_csv(summary_path, index=False)


def run_datasets(names: list[str], prepare_only: bool = False) -> list[dict]:
    """Prepare and evaluate datasets selected only by their names."""
    results = []
    for requested_name in names:
        name, spec, data, source_run = prepare_dataset(requested_name)
        print(f"Prepared {name}: {data}", flush=True)
        if prepare_only:
            continue
        if spec["task"] == "binary":
            result = run_binary_dataset(data)
            update_binary_summary(result)
        else:
            if source_run is None:
                raise RuntimeError(f"Missing cross-attention run for {name}")
            result = run_multilabel_dataset(
                data, source_run, runtime_key=spec.get("runtime_key"))
            update_multilabel_summary(result)
        results.append(result)
    return results

def command_line() -> None:
    """Evaluate existing HyCL-ACP training artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained HyCL-ACP artifacts")
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    args = parser.parse_args()
    run_datasets(args.datasets)


if __name__ == "__main__":
    command_line()

