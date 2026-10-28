"""
Comprehensive unit and integration test script for Software Defect Prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data_loader import load_raw_dataset, get_dataset_inventory, METRIC_DEFINITIONS
from src.preprocessing import clean_software_dataset, split_dataset, get_imputed_dataframe
from src.eda import (
    plot_defect_distribution,
    plot_feature_distribution,
    plot_box_comparison,
    plot_correlation_heatmap,
    plot_scatter_relationship,
    plot_defect_rate_by_bins,
    compute_group_statistics
)
from src.model_training import (
    train_defect_models,
    save_trained_pipeline,
    load_saved_pipeline
)
from src.evaluation import (
    evaluate_pipeline,
    plot_confusion_matrix,
    plot_model_comparison_bars,
    build_comparison_dataframe
)
from src.feature_analysis import (
    get_decision_tree_importance,
    get_svm_permutation_importance
)
from src.prediction import predict_defect_risk, validate_module_inputs
from src.recommendations import generate_qa_recommendations, get_academic_limitations


def run_all_tests():
    print("=" * 60)
    print("RUNNING SOFTWARE DEFECT PREDICTION SUITE VERIFICATION")
    print("=" * 60)

    # 1. Inventory Check
    print("\n[TEST 1] Checking Dataset Inventory...")
    inv = get_dataset_inventory()
    print("Inventory status:", inv)
    assert inv["cm1"]["available"], "CM1 file should be available."
    assert inv["jm1"]["available"], "JM1 file should be available."
    print("--> PASS: Both CM1 and JM1 files exist.")

    # 2. Test CM1 & JM1 Loading and Preprocessing
    for name in ["cm1", "jm1"]:
        print(f"\n[TEST 2] Testing loading & preprocessing on {name.upper()}...")
        df_raw, err = load_raw_dataset(name)
        assert err is None, f"Loading error on {name}: {err}"
        assert df_raw is not None, f"Dataset {name} should not be None"
        print(f"Raw shape: {df_raw.shape}")

        df_clean, summary = clean_software_dataset(df_raw, dataset_name=name)
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Defect rate: {summary.defect_rate_percent}%")
        print(f"Converted columns: {summary.converted_columns}")
        print(f"Missing values after conversion: {summary.missing_values_original}")
        print(f"Duplicate metric profiles: {summary.duplicate_metric_profiles}")

        # Check target is boolean
        assert df_clean["defects"].dtype == bool or np.issubdtype(df_clean["defects"].dtype, np.bool_), "Defects must be boolean"
        # Check id is preserved
        assert "id" in df_clean.columns, "id column must be preserved in clean_df"

        # Check leak-free split
        feature_cols = summary.feature_columns
        assert "id" not in feature_cols, "'id' must NOT be in feature columns!"
        assert "defects" not in feature_cols, "'defects' must NOT be in feature columns!"

        X_train, X_test, y_train, y_test, meta = split_dataset(
            df_clean, feature_cols, target_col="defects", test_size=0.20, random_state=42
        )
        print(f"Split: Train {len(X_train)} samples, Test {len(X_test)} samples. Stratified: {meta['stratified']}")
        assert len(X_train) + len(X_test) == len(df_clean)
        print(f"--> PASS: Preprocessing and split verified on {name.upper()}.")

    # 3. Test EDA functions on CM1
    print("\n[TEST 3] Testing EDA generation...")
    df_cm1, _ = load_raw_dataset("cm1")
    df_clean_cm1, sum_cm1 = clean_software_dataset(df_cm1, "cm1")
    fig_dist = plot_defect_distribution(df_clean_cm1, "cm1")
    assert fig_dist is not None
    fig_feat = plot_feature_distribution(df_clean_cm1, "loc", "Lines of Code")
    assert fig_feat is not None
    fig_box = plot_box_comparison(df_clean_cm1, "v(g)", "Cyclomatic Complexity")
    assert fig_box is not None
    fig_corr = plot_correlation_heatmap(df_clean_cm1, sum_cm1.feature_columns)
    assert fig_corr is not None
    fig_scatter = plot_scatter_relationship(df_clean_cm1, "loc", "v(g)")
    assert fig_scatter is not None
    fig_rate, stats = plot_defect_rate_by_bins(df_clean_cm1, "loc", num_bins=4)
    assert fig_rate is not None
    stats_df = compute_group_statistics(df_clean_cm1, sum_cm1.feature_columns)
    assert not stats_df.empty
    print("--> PASS: All EDA visualizations generated successfully.")

    # 4. Test Model Training on CM1
    print("\n[TEST 4] Testing Model Training (Decision Tree & SVM) on CM1...")
    feature_cols = sum_cm1.feature_columns
    X_train, X_test, y_train, y_test, meta = split_dataset(
        df_clean_cm1, feature_cols, test_size=0.20, random_state=42
    )
    models = train_defect_models(X_train, y_train)
    assert "Decision Tree" in models
    assert "Support Vector Machine" in models
    print("--> PASS: Both models trained successfully.")

    # 5. Test Evaluation & Comparison
    print("\n[TEST 5] Testing Evaluation on held-out test data...")
    eval_results = {}
    for m_name, pipe in models.items():
        res = evaluate_pipeline(pipe, X_test, y_test, dataset_split="Test")
        eval_results[m_name] = res
        print(f"  {m_name}: Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, F1={res['f1_score']:.4f}, ROC-AUC={res['roc_auc']}")
        assert res["accuracy"] > 0
        assert "confusion_matrix" in res

    comp_df = build_comparison_dataframe(eval_results)
    print("Comparison DataFrame:\n", comp_df)
    fig_bars = plot_model_comparison_bars(eval_results)
    assert fig_bars is not None
    print("--> PASS: Evaluation metrics and comparison tables verified.")

    # 6. Test Feature Analysis
    print("\n[TEST 6] Testing Feature Importance & Permutation Importance...")
    dt_imp = get_decision_tree_importance(models["Decision Tree"], feature_cols)
    print("Top 3 Decision Tree features:\n", dt_imp[["Metric", "Importance_Pct", "Family"]].head(3))
    assert not dt_imp.empty

    svm_perm = get_svm_permutation_importance(models["Support Vector Machine"], X_test, y_test, feature_cols, n_repeats=3)
    print("Top 3 SVM Permutation features:\n", svm_perm[["Metric", "Importance_Pct", "Family"]].head(3))
    assert not svm_perm.empty
    print("--> PASS: Feature analysis verified.")

    # 7. Test Prediction on Single Module
    print("\n[TEST 7] Testing Interactive Single-Module Prediction...")
    # Generate test input from median values
    sample_input = {feat: float(X_test[feat].median()) for feat in feature_cols}
    # Test valid input
    is_valid, msg = validate_module_inputs(sample_input, feature_cols)
    assert is_valid, f"Validation should succeed: {msg}"

    pred_res_dt = predict_defect_risk(models["Decision Tree"], sample_input, feature_cols)
    print(f"Decision Tree Prediction: {pred_res_dt['status']} (Risk: {pred_res_dt['risk_percentage']}%, Tier: {pred_res_dt['risk_tier']})")
    assert pred_res_dt["status"] in ["Defective", "Defect-Free"]

    pred_res_svm = predict_defect_risk(models["Support Vector Machine"], sample_input, feature_cols)
    print(f"SVM Prediction: {pred_res_svm['status']} (Risk: {pred_res_svm['risk_percentage']}%, Tier: {pred_res_svm['risk_tier']})")
    assert pred_res_svm["status"] in ["Defective", "Defect-Free"]
    print("--> PASS: Single-module prediction verified.")

    # 8. Test Persistence (Save and Load)
    print("\n[TEST 8] Testing Model Persistence (joblib + metadata JSON)...")
    m_path, meta_path = save_trained_pipeline(
        models["Decision Tree"],
        model_name="Decision Tree",
        dataset_name="cm1",
        feature_names=feature_cols,
        evaluation_metrics=eval_results["Decision Tree"]
    )
    print(f"Saved to: {m_path} and {meta_path}")

    loaded_pipe, loaded_meta, err = load_saved_pipeline("Decision Tree", "cm1")
    assert err is None, f"Failed to load saved model: {err}"
    assert loaded_pipe is not None
    assert loaded_meta["dataset_name"] == "cm1"
    print("--> PASS: Model save and load successfully verified.")

    # 9. Test Recommendations
    print("\n[TEST 9] Testing Recommendations...")
    recs = generate_qa_recommendations("cm1", df_clean_cm1, eval_results)
    assert len(recs) > 0
    print(f"Generated {len(recs)} QA recommendations.")
    limits = get_academic_limitations()
    assert len(limits) > 0
    print(f"Retrieved {len(limits)} academic project limitations.")
    print("--> PASS: Recommendations verified.")

    print("\n" + "=" * 60)
    print("ALL CORE UNIT AND INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
