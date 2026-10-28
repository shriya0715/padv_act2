"""
Insights and Recommendations Module for Software Defect Prediction.

Synthesizes data distribution properties, feature associations, and model performance
to generate actionable quality-assurance recommendations and academic project limitations.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


def generate_qa_recommendations(
    dataset_name: str,
    df: pd.DataFrame,
    eval_results: Optional[Dict[str, Dict[str, Any]]] = None,
    top_features: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generates actionable QA and testing recommendations conditioned on actual data and model outcomes.
    """
    recommendations = []

    # 1. Complexity-driven inspection
    if "v(g)" in df.columns:
        high_vg_count = int((df["v(g)"] > 10).sum())
        high_vg_pct = round(high_vg_count / len(df) * 100, 1)
        if "defects" in df.columns:
            vg_defect_rate = round(float(df[df["v(g)"] > 10]["defects"].mean() * 100), 1)
            baseline_rate = round(float(df["defects"].mean() * 100), 1)
        else:
            vg_defect_rate = 0.0
            baseline_rate = 0.0

        recommendations.append({
            "category": "Architectural Refactoring & Complexity Control",
            "priority": "High" if high_vg_pct > 15 else "Medium",
            "title": "Enforce McCabe Cyclomatic Complexity Gates (v(g) <= 10)",
            "observation": (
                f"{high_vg_count:,} modules ({high_vg_pct}%) exhibit cyclomatic complexity v(g) > 10. "
                f"Modules in this range demonstrate an observed defect rate of {vg_defect_rate}% "
                f"(compared to the {baseline_rate}% baseline)."
            ),
            "action": (
                "Establish automated CI/CD linting gates blocking pull requests where new or modified functions "
                "exceed v(g)=10. Decompose multi-branch procedures into modular helper functions."
            )
        })

    # 2. Module Size & Monolith Management
    if "loc" in df.columns:
        large_loc_count = int((df["loc"] > 100).sum())
        large_loc_pct = round(large_loc_count / len(df) * 100, 1)
        recommendations.append({
            "category": "Code Review & Sizing",
            "priority": "Medium",
            "title": "Target Monolithic Modules for Refactoring (LOC > 100)",
            "observation": (
                f"{large_loc_count:,} modules ({large_loc_pct}%) exceed 100 lines of code. "
                "Empirical software engineering studies show large modules experience higher maintenance friction."
            ),
            "action": (
                "Prioritize peer code reviews and modular refactoring for modules with LOC > 100, "
                "extracting cohesive subroutines and applying Single Responsibility Principle."
            )
        })

    # 3. Model-driven triage & false negative mitigation
    if eval_results:
        best_model_name = None
        best_recall = -1.0
        best_f1 = -1.0
        for name, metrics in eval_results.items():
            f1_score = metrics.get("f1_score", 0.0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_name = name
                best_recall = metrics.get("recall", 0.0)

        dt_res = eval_results.get("Decision Tree", {})
        svm_res = eval_results.get("Support Vector Machine", {})
        fn_count = svm_res.get("confusion_matrix", {}).get("fn", 0)

        recommendations.append({
            "category": "Quality Assurance Resource Allocation",
            "priority": "Critical",
            "title": f"Deploy {best_model_name} for Risk-Based Testing Prioritization",
            "observation": (
                f"{best_model_name} achieved the strongest balance on held-out test data (F1 = {best_f1:.3f}, "
                f"Recall = {best_recall:.3f}). However, test evaluation revealed {fn_count} false negative(s) "
                "where defective modules escaped detection."
            ),
            "action": (
                "Use model predictions to rank test case execution order (focusing exploratory testing and "
                "mutation testing on high-risk modules first). Never treat a defect-free prediction as an excuse "
                "to skip core unit tests."
            )
        })

    # 4. Halstead Effort & Comprehension Load
    if "e" in df.columns:
        recommendations.append({
            "category": "Cognitive Complexity & Documentation",
            "priority": "Low",
            "title": "Enhance Unit Test Coverage for High Halstead Effort Modules",
            "observation": (
                "Halstead Effort (e) combines volume and mental difficulty. High-effort modules "
                "require disproportionate cognitive load during maintenance."
            ),
            "action": (
                "Require minimum 85% branch coverage and descriptive inline documentation for the top 10% "
                "highest-effort modules."
            )
        })

    return recommendations


def get_academic_limitations() -> List[Dict[str, str]]:
    """
    Returns the analytical limitations for academic defense, thesis reports, and project presentations.
    """
    return [
        {
            "aspect": "Dataset Age and Historical Context",
            "description": (
                "The CM1 and JM1 datasets originate from the NASA Metrics Data Program (MDP) collected in the late 1990s and early 2000s. "
                "They reflect procedural and object-oriented C/C++ aerospace software and may not fully generalize to modern high-level languages, "
                "microservices architectures, or contemporary agile CI/CD practices."
            )
        },
        {
            "aspect": "Severe Class Imbalance",
            "description": (
                "Both datasets exhibit class imbalance (CM1 has ~9.8% defective modules; JM1 has ~19.4%). Standard accuracy can be "
                "misleading because a naive classifier predicting all modules are defect-free achieves 90% accuracy while failing completely. "
                "Evaluation must focus on Recall, Precision, and F1-score with appropriate class weighting."
            )
        },
        {
            "aspect": "Correlation vs. Causation",
            "description": (
                "Static software metrics (such as LOC or cyclomatic complexity) have predictive associations with defect presence, "
                "but high complexity does not mechanically cause bugs. Developer fatigue, requirement ambiguity, and architectural changes "
                "are latent root causes not captured by static metric counters."
            )
        },
        {
            "aspect": "Cross-Project Generalization",
            "description": (
                "CM1 (instrument spacecraft subsystem) and JM1 (real-time ground tracking system) differ substantially in module volume, "
                "coding styles, and defect density. A model trained on CM1 cannot be applied blindly to JM1 without domain adaptation."
            )
        },
        {
            "aspect": "False Negatives and Safety Risks",
            "description": (
                "In safety-critical domains, a false negative (predicting a defective module is defect-free) is vastly more perilous "
                "than a false positive (a false alarm prompting an extra code review). Model thresholds should be calibrated accordingly."
            )
        }
    ]
