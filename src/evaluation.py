"""
Model Evaluation and Comparison Module for Software Defect Prediction.

Calculates comprehensive classification metrics, generates confusion matrices,
ROC and PR curves, interactive comparison charts, and plain-language metric explanations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline

# Metric definitions in plain language for student project reports and tooltips
METRIC_EXPLANATIONS = {
    "accuracy": "Proportion of total module predictions that were correct (both defective and defect-free). Can be misleading under class imbalance.",
    "precision": "Proportion of predicted defective modules that were truly defective. High precision means fewer false alarms.",
    "recall": "Proportion of actual defective modules successfully identified by the model. Critical for QA to prevent buggy code reaching production.",
    "f1_score": "Harmonic mean of precision and recall. Provides a balanced metric especially when defective modules are a minority class.",
    "roc_auc": "Area Under the Receiver Operating Characteristic Curve. Evaluates discrimination across all possible decision thresholds (0.5 = random, 1.0 = perfect).",
    "avg_precision": "Area Under the Precision-Recall Curve. Highly informative for imbalanced software defect datasets."
}


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    dataset_split: str = "Test"
) -> Dict[str, Any]:
    """
    Computes all classification metrics for a trained pipeline on a given dataset split.
    Uses positive class = 1 (Defective module).
    """
    y_true = np.array(y).astype(int)
    y_pred = pipeline.predict(X)

    # Compute probabilities if supported
    y_proba = None
    has_proba = False
    if hasattr(pipeline, "predict_proba"):
        try:
            proba_all = pipeline.predict_proba(X)
            # Probability of defective class (index 1)
            if proba_all.shape[1] > 1:
                y_proba = proba_all[:, 1]
                has_proba = True
        except Exception:
            y_proba = None

    # Calculate metrics with zero_division handling
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # ROC-AUC and Average Precision
    roc_auc = None
    avg_prec = None
    roc_data = None
    pr_data = None

    if has_proba and y_proba is not None:
        try:
            # Check if both classes are present in y_true
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_proba))
                avg_prec = float(average_precision_score(y_true, y_proba))
                
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
                roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

                p_curve, r_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)
                pr_data = {"precision": p_curve.tolist(), "recall": r_curve.tolist()}
        except Exception:
            roc_auc = None
            avg_prec = None

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Detailed report dict
    report_dict = classification_report(
        y_true, y_pred,
        target_names=["Defect-Free (0)", "Defective (1)"],
        output_dict=True,
        zero_division=0
    )

    return {
        "split": dataset_split,
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "avg_precision": round(avg_prec, 4) if avg_prec is not None else None,
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "raw": cm.tolist()
        },
        "classification_report": report_dict,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist() if y_proba is not None else None
    }


def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str, split_name: str = "Test") -> go.Figure:
    """
    Renders an interactive confusion matrix heatmap with cell counts, percentages, and labels.
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    total = tn + fp + fn + tp

    z = [[tn, fp], [fn, tp]]
    
    # Text annotations with rates
    annotations = [
        [
            f"<b>True Negative (TN)</b><br>{tn:,}<br>({round(tn/total*100, 1)}%)",
            f"<b>False Positive (FP)</b><br>{fp:,}<br>({round(fp/total*100, 1)}%)<br><i>False Alarm</i>"
        ],
        [
            f"<b>False Negative (FN)</b><br>{fn:,}<br>({round(fn/total*100, 1)}%)<br><i>Escaped Bug</i>",
            f"<b>True Positive (TP)</b><br>{tp:,}<br>({round(tp/total*100, 1)}%)<br><i>Caught Defect</i>"
        ]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Defect-Free", "Predicted Defective"],
        y=["Actual Defect-Free", "Actual Defective"],
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 13, "family": "Inter, sans-serif"},
        colorscale=[[0, "#EBF3FB"], [1, "#1D3557"]],
        showscale=False,
        hoverinfo="x+y+z"
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix: {model_name} ({split_name} Set)</b>",
        xaxis=dict(title="<b>Predicted Label</b>", side="bottom"),
        yaxis=dict(title="<b>Ground Truth Label</b>", autorange="reversed"),
        height=380,
        margin=dict(l=60, r=40, t=60, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    return fig


def plot_model_comparison_bars(eval_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing Decision Tree and SVM on Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = list(eval_results.keys())

    fig = go.Figure()

    colors = ["#1D3557", "#2A9D8F", "#E76F51"]
    for i, model in enumerate(models):
        res = eval_results[model]
        vals = [
            res.get("accuracy", 0.0),
            res.get("precision", 0.0),
            res.get("recall", 0.0),
            res.get("f1_score", 0.0),
            res.get("roc_auc") if res.get("roc_auc") is not None else 0.0
        ]
        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=vals,
            text=[f"{v:.3f}" if v is not None else "N/A" for v in vals],
            textposition="outside",
            marker_color=color
        ))

    fig.update_layout(
        title="<b>Model Performance Metric Comparison (Test Set)</b>",
        yaxis=dict(title="Score", range=[0, 1.15], showgrid=True, gridcolor="#E5E5E5"),
        barmode="group",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_roc_curves(eval_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Plots ROC curves for models having predicted probability outputs.
    """
    fig = go.Figure()

    # Diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.50)",
        line=dict(color="#94A3B8", dash="dash")
    ))

    colors = ["#1D3557", "#2A9D8F", "#E63946"]
    for i, (model_name, res) in enumerate(eval_results.items()):
        roc = res.get("roc_curve")
        auc_val = res.get("roc_auc")
        if roc and roc.get("fpr") and roc.get("tpr"):
            label = f"{model_name} (AUC = {auc_val:.3f})" if auc_val else model_name
            fig.add_trace(go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name=label,
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], showgrid=True),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[-0.02, 1.05], showgrid=True),
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_precision_recall_curves(
    eval_results: Dict[str, Dict[str, Any]],
    baseline_defect_rate: float = 0.10
) -> go.Figure:
    """
    Plots Precision-Recall curves alongside the dataset baseline defect rate.
    """
    fig = go.Figure()

    # Baseline horizontal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline_defect_rate, baseline_defect_rate],
        mode="lines",
        name=f"Baseline Defect Rate ({round(baseline_defect_rate * 100, 1)}%)",
        line=dict(color="#94A3B8", dash="dash")
    ))

    colors = ["#1D3557", "#2A9D8F", "#E63946"]
    for i, (model_name, res) in enumerate(eval_results.items()):
        pr = res.get("pr_curve")
        ap_val = res.get("avg_precision")
        if pr and pr.get("recall") and pr.get("precision"):
            label = f"{model_name} (Avg Prec = {ap_val:.3f})" if ap_val else model_name
            fig.add_trace(go.Scatter(
                x=pr["recall"],
                y=pr["precision"],
                mode="lines",
                name=label,
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))

    fig.update_layout(
        title="<b>Precision-Recall (PR) Curves</b>",
        xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02], showgrid=True),
        yaxis=dict(title="Precision", range=[-0.02, 1.05], showgrid=True),
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def build_comparison_dataframe(eval_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a clean tabular comparison of model evaluation metrics.
    """
    rows = []
    for model_name, res in eval_results.items():
        cm = res.get("confusion_matrix", {})
        rows.append({
            "Model": model_name,
            "Accuracy": f"{res.get('accuracy', 0):.4f}",
            "Precision": f"{res.get('precision', 0):.4f}",
            "Recall": f"{res.get('recall', 0):.4f}",
            "F1-Score": f"{res.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{res.get('roc_auc', 0):.4f}" if res.get("roc_auc") is not None else "N/A",
            "Avg Precision": f"{res.get('avg_precision', 0):.4f}" if res.get("avg_precision") is not None else "N/A",
            "Caught Defects (TP)": cm.get("tp", 0),
            "Missed Defects (FN)": cm.get("fn", 0),
            "False Alarms (FP)": cm.get("fp", 0),
            "Correct Clean (TN)": cm.get("tn", 0),
        })
    return pd.DataFrame(rows)
