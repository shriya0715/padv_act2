"""
Single-Module Defect Prediction Module for Software Defect Prediction.

Handles real-time feature validation, alignment, model scoring,
probability estimation, risk tiering, and contextual diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def validate_module_inputs(
    input_data: Dict[str, float],
    feature_names: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validates user-provided software metric values.
    Returns (is_valid, list_of_error_or_warning_messages).
    """
    messages = []
    is_valid = True

    # Check for missing required features
    missing_feats = [f for f in feature_names if f not in input_data]
    if missing_feats:
        is_valid = False
        messages.append(f"Missing required metric(s): {', '.join(missing_feats)}")

    # Check for negative metric values (counts and sizes cannot be negative)
    for feat, val in input_data.items():
        if val is None or np.isnan(val):
            messages.append(f"Metric '{feat}' contains an empty or NaN value; default imputation will apply.")
        elif val < 0:
            is_valid = False
            messages.append(f"Metric '{feat}' cannot be negative (provided: {val}).")

    return is_valid, messages


def predict_defect_risk(
    pipeline: Pipeline,
    input_data: Dict[str, float],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Runs single-module inference using a trained scikit-learn Pipeline.
    
    Parameters:
        pipeline: Trained Decision Tree or SVM Pipeline.
        input_data: Mapping of metric names to user-entered values.
        feature_names: Exact ordered list of features used during training.
        
    Returns:
        Dictionary containing prediction results, probability, risk level, and diagnostics.
    """
    # Align features exactly in training order
    row_values = [input_data.get(feat, np.nan) for feat in feature_names]
    input_df = pd.DataFrame([row_values], columns=feature_names)

    # Prediction
    pred_label = int(pipeline.predict(input_df)[0])
    status_str = "Defective" if pred_label == 1 else "Defect-Free"

    # Probability extraction
    probability_defective = None
    probability_defect_free = None
    has_probability = False

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(input_df)[0]
            if len(probs) >= 2:
                probability_defect_free = float(probs[0])
                probability_defective = float(probs[1])
                has_probability = True
            elif len(probs) == 1:
                # Single class edge case
                probability_defective = float(pred_label)
                probability_defect_free = 1.0 - probability_defective
                has_probability = True
        except Exception:
            has_probability = False

    # Risk level categorization
    if has_probability and probability_defective is not None:
        risk_pct = round(probability_defective * 100, 1)
        if risk_pct >= 60.0:
            risk_tier = "High Risk"
            risk_badge_color = "#E63946"
        elif risk_pct >= 35.0:
            risk_tier = "Moderate Risk"
            risk_badge_color = "#F4A261"
        else:
            risk_tier = "Low Risk"
            risk_badge_color = "#2A9D8F"
    else:
        risk_pct = 100.0 if pred_label == 1 else 0.0
        risk_tier = "High Risk" if pred_label == 1 else "Low Risk"
        risk_badge_color = "#E63946" if pred_label == 1 else "#2A9D8F"

    # Contextual diagnostic guidance
    loc_val = input_data.get("loc", 0)
    vg_val = input_data.get("v(g)", 0)
    branch_val = input_data.get("branchCount", 0)
    effort_val = input_data.get("e", 0)

    flags = []
    if vg_val > 10:
        flags.append(f"Cyclomatic complexity v(g)={vg_val:.1f} exceeds McCabe's standard threshold of 10.")
    if loc_val > 100:
        flags.append(f"Lines of code loc={loc_val:.1f} indicates a relatively large, monolithic module.")
    if branch_val > 15:
        flags.append(f"Branch count={branch_val:.1f} suggests numerous nested conditional pathways.")
    if effort_val > 5000:
        flags.append(f"Halstead effort e={effort_val:,.0f} reflects elevated cognitive comprehension load.")

    return {
        "status": status_str,
        "is_defective": bool(pred_label == 1),
        "predicted_label": pred_label,
        "probability_defective": probability_defective,
        "probability_defect_free": probability_defect_free,
        "risk_percentage": risk_pct,
        "risk_tier": risk_tier,
        "risk_color": risk_badge_color,
        "has_probability": has_probability,
        "input_features": {feat: input_data.get(feat) for feat in feature_names},
        "diagnostic_flags": flags,
        "disclaimer": (
            "Notice: Predictions are statistical estimates derived from historical NASA MDP software metrics. "
            "A defect prediction does not guarantee bug presence, and a defect-free prediction does not replace "
            "rigorous unit testing and systematic code review."
        )
    }
