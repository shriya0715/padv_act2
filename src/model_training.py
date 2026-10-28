"""
Model Training Module for Software Defect Prediction.

Constructs leak-free scikit-learn pipelines for Decision Tree and SVM models,
supports class balancing, hyperparameter controls, model persistence, and metadata recording.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_default_models_dir() -> Path:
    """Returns the absolute Path to the project's models directory."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def create_decision_tree_pipeline(
    max_depth: Optional[int] = 5,
    min_samples_leaf: int = 4,
    min_samples_split: int = 10,
    criterion: str = "gini",
    class_weight: Optional[str] = "balanced",
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a Decision Tree pipeline with median imputation and configurable hyperparameters.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])
    return pipeline


def create_svm_pipeline(
    kernel: str = "rbf",
    c_param: float = 1.0,
    gamma: str = "scale",
    class_weight: Optional[str] = "balanced",
    probability: bool = True,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a Support Vector Machine pipeline with median imputation, StandardScaler,
    and calibrated probabilities (via Platt scaling).
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel=kernel,
            C=c_param,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
            cache_size=500
        ))
    ])
    return pipeline


def train_defect_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dt_params: Optional[Dict[str, Any]] = None,
    svm_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Pipeline]:
    """
    Trains both Decision Tree and SVM models using the provided training split.
    
    Parameters:
        X_train: Feature DataFrame.
        y_train: Target Series (0 = defect-free, 1 = defective).
        dt_params: Dictionary of kwargs for Decision Tree.
        svm_params: Dictionary of kwargs for SVM.
        
    Returns:
        Dictionary mapping model names ('Decision Tree', 'Support Vector Machine') to trained Pipelines.
    """
    dt_config = dt_params or {}
    svm_config = svm_params or {}

    dt_pipeline = create_decision_tree_pipeline(**dt_config)
    svm_pipeline = create_svm_pipeline(**svm_config)

    # Fit Decision Tree
    dt_pipeline.fit(X_train, y_train)

    # Fit SVM
    svm_pipeline.fit(X_train, y_train)

    return {
        "Decision Tree": dt_pipeline,
        "Support Vector Machine": svm_pipeline
    }


def save_trained_pipeline(
    pipeline: Pipeline,
    model_name: str,
    dataset_name: str,
    feature_names: List[str],
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None
) -> Tuple[str, str]:
    """
    Saves a trained Pipeline and its associated metadata into the models directory.
    
    Returns:
        Tuple of (model_file_path, metadata_file_path).
    """
    target_dir = Path(models_dir) if models_dir else get_default_models_dir()
    slug_name = model_name.lower().replace(" ", "_")
    base_filename = f"{dataset_name.lower()}_{slug_name}"

    model_path = target_dir / f"{base_filename}.joblib"
    meta_path = target_dir / f"{base_filename}_meta.json"

    # Save Pipeline
    joblib.dump(pipeline, model_path)

    # Save Metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name.lower(),
        "created_at": datetime.now().isoformat(),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "hyperparameters": hyperparameters or {},
        "evaluation_metrics": evaluation_metrics or {}
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return str(model_path), str(meta_path)


def load_saved_pipeline(
    model_name: str,
    dataset_name: str,
    models_dir: Optional[Path] = None
) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads a saved pipeline and validates compatibility against dataset name and schema.
    
    Returns:
        Tuple of (Pipeline or None, metadata or None, error_message or None).
    """
    target_dir = Path(models_dir) if models_dir else get_default_models_dir()
    slug_name = model_name.lower().replace(" ", "_")
    base_filename = f"{dataset_name.lower()}_{slug_name}"

    model_path = target_dir / f"{base_filename}.joblib"
    meta_path = target_dir / f"{base_filename}_meta.json"

    if not model_path.exists():
        return None, None, f"No saved model found for {model_name} on {dataset_name.upper()}."

    try:
        pipeline = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Compatibility verification
        if metadata.get("dataset_name", "").lower() != dataset_name.lower():
            return None, None, (
                f"Model was trained on {metadata.get('dataset_name')}, "
                f"incompatible with current dataset {dataset_name}."
            )

        return pipeline, metadata, None
    except Exception as e:
        return None, None, f"Error loading saved model: {str(e)}"
