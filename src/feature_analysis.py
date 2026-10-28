"""
Feature Analysis Module for Software Defect Prediction.

Calculates and visualizes:
1. Gini Feature Importance for Decision Tree classifiers.
2. Permutation Feature Importance on held-out test data for SVM classifiers.
3. Plain-language interpretations distinguishing predictive correlation from causation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import METRIC_DEFINITIONS


def get_decision_tree_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extracts Gini-based feature importance from a trained Decision Tree pipeline.
    """
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, DecisionTreeClassifier):
        raise ValueError("Pipeline classifier is not a DecisionTreeClassifier.")

    importances = classifier.feature_importances_
    df = pd.DataFrame({
        "Metric": feature_names,
        "Importance": importances,
        "Importance_Pct": [round(val * 100, 2) for val in importances]
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Attach descriptive labels and families
    df["Description"] = df["Metric"].apply(
        lambda m: METRIC_DEFINITIONS.get(m, {}).get("description", "Software metric")
    )
    df["Family"] = df["Metric"].apply(
        lambda m: METRIC_DEFINITIONS.get(m, {}).get("family", "Other")
    )
    return df


def get_svm_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    n_repeats: int = 3,
    random_state: int = 42,
    scoring: str = "f1",
    max_samples: int = 500
) -> pd.DataFrame:
    """
    Calculates permutation importance for SVM on held-out test data.
    Uses representative sampling if test set exceeds max_samples for fast interactive UI responsiveness.
    Note: RBF SVM does not have linear coefficients; permutation importance measures
    how much the model score drops when feature values are randomly shuffled.
    """
    if len(X_test) > max_samples:
        eval_X = X_test.sample(n=max_samples, random_state=random_state)
        eval_y = y_test.loc[eval_X.index]
    else:
        eval_X = X_test
        eval_y = y_test

    perm = permutation_importance(
        pipeline,
        eval_X,
        eval_y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1
    )

    df = pd.DataFrame({
        "Metric": feature_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
        "Importance_Pct": [round(max(0.0, val) * 100, 2) for val in perm.importances_mean]
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    df["Description"] = df["Metric"].apply(
        lambda m: METRIC_DEFINITIONS.get(m, {}).get("description", "Software metric")
    )
    df["Family"] = df["Metric"].apply(
        lambda m: METRIC_DEFINITIONS.get(m, {}).get("family", "Other")
    )
    return df


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 10,
    metric_col: str = "Importance"
) -> go.Figure:
    """
    Creates a clean horizontal bar chart of the top N most influential features.
    """
    display_df = importance_df.head(top_n).sort_values(by=metric_col, ascending=True).copy()

    fig = go.Figure(go.Bar(
        x=display_df[metric_col],
        y=display_df["Metric"],
        orientation="h",
        marker=dict(
            color=display_df[metric_col],
            colorscale=[[0, "#93C5FD"], [1, "#1D3557"]],
            showscale=False
        ),
        text=[f"{v:.4f}" for v in display_df[metric_col]],
        textposition="outside",
        hovertext=[
            f"<b>{m}</b> ({f})<br>{d}<br>Score: {v:.4f}"
            for m, f, d, v in zip(
                display_df["Metric"],
                display_df["Family"],
                display_df["Description"],
                display_df[metric_col]
            )
        ],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"<b>{title} (Top {min(top_n, len(importance_df))} Metrics)</b>",
        xaxis=dict(title="Importance Weight / Score Drop", showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(title="Software Metric", automargin=True),
        template="plotly_white",
        height=max(380, top_n * 28),
        margin=dict(l=80, r=40, t=50, b=40)
    )
    return fig
