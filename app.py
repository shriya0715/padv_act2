"""
Software Defect Prediction System
Interactive Machine Learning & Analytics Dashboard
Built with Streamlit, Scikit-Learn, and Plotly
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import core modules
from src.data_loader import (
    load_raw_dataset,
    get_dataset_inventory,
    METRIC_DEFINITIONS
)
from src.preprocessing import (
    clean_software_dataset,
    split_dataset,
    get_imputed_dataframe,
    PreprocessingSummary
)
from src.eda import (
    plot_defect_distribution,
    plot_feature_distribution,
    plot_box_comparison,
    plot_correlation_heatmap,
    plot_scatter_relationship,
    plot_defect_rate_by_bins,
    compute_group_statistics,
    plot_dataset_comparison_chart
)
from src.model_training import (
    train_defect_models,
    save_trained_pipeline,
    load_saved_pipeline,
    get_default_models_dir
)
from src.evaluation import (
    evaluate_pipeline,
    plot_confusion_matrix,
    plot_model_comparison_bars,
    plot_roc_curves,
    plot_precision_recall_curves,
    build_comparison_dataframe,
    METRIC_EXPLANATIONS
)
from src.feature_analysis import (
    get_decision_tree_importance,
    get_svm_permutation_importance,
    plot_feature_importance_bar
)
from src.prediction import (
    predict_defect_risk,
    validate_module_inputs
)
from src.recommendations import (
    generate_qa_recommendations,
    get_academic_limitations
)


# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Software Defect Prediction Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Font and Container Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    .main-header {
        background: linear-gradient(135deg, #1D3557 0%, #2A4365 100%);
        color: #FFFFFF;
        padding: 24px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(29, 53, 87, 0.12);
    }
    .main-header h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 14px;
        color: #E2E8F0;
        margin: 6px 0 0 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 4px;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #94A3B8;
    }
    
    /* Risk Badges */
    .badge-defect {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .badge-clean {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    
    /* Action & Recommendation Cards */
    .rec-card {
        background-color: #FFFFFF;
        border-left: 4px solid #1D3557;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .rec-card.high-priority {
        border-left-color: #E63946;
    }
    .rec-card.medium-priority {
        border-left-color: #F4A261;
    }
    .rec-card.low-priority {
        border-left-color: #2A9D8F;
    }
    
    /* Disclaimers & Notes */
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 13px;
        color: #475569;
        line-height: 1.5;
        margin-top: 16px;
    }
    
    /* Clean Tab and Sidebar elements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Caching Functions for Data & Models
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_raw_data(dataset_name: str):
    """Loads and returns raw dataframe from CSV."""
    return load_raw_dataset(dataset_name)


@st.cache_data(show_spinner=False)
def get_cached_cleaned_data(dataset_name: str):
    """Loads, cleans, and summarizes the dataset."""
    df_raw, err = load_raw_dataset(dataset_name)
    if err or df_raw is None:
        return None, None, err
    clean_df, summary = clean_software_dataset(df_raw, dataset_name)
    return clean_df, summary, None


def ensure_models_trained(
    clean_df: pd.DataFrame,
    summary: PreprocessingSummary,
    dataset_name: str,
    test_size: float = 0.20,
    force_retrain: bool = False,
    dt_params: Optional[Dict[str, Any]] = None,
    svm_params: Optional[Dict[str, Any]] = None
):
    """
    Checks session_state for trained models compatible with dataset_name.
    Trains or loads if not present or if force_retrain is True.
    """
    state_key = f"models_{dataset_name.lower()}"
    eval_key = f"eval_{dataset_name.lower()}"
    split_key = f"split_{dataset_name.lower()}"

    if (
        not force_retrain
        and state_key in st.session_state
        and eval_key in st.session_state
        and split_key in st.session_state
    ):
        return (
            st.session_state[state_key],
            st.session_state[eval_key],
            st.session_state[split_key]
        )

    # Perform stratified split
    feature_cols = summary.feature_columns
    X_train, X_test, y_train, y_test, meta = split_dataset(
        clean_df,
        feature_cols,
        target_col="defects",
        test_size=test_size,
        random_state=42,
        stratify=True
    )

    # Train Decision Tree and SVM models
    models = train_defect_models(
        X_train,
        y_train,
        dt_params=dt_params,
        svm_params=svm_params
    )

    # Evaluate models on held-out test split
    eval_results = {}
    for m_name, pipeline in models.items():
        eval_results[m_name] = evaluate_pipeline(
            pipeline,
            X_test,
            y_test,
            dataset_split="Test"
        )
        # Persist model to disk with metadata
        try:
            save_trained_pipeline(
                pipeline=pipeline,
                model_name=m_name,
                dataset_name=dataset_name,
                feature_names=feature_cols,
                evaluation_metrics=eval_results[m_name]
            )
        except Exception:
            pass

    split_payload = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "meta": meta
    }

    st.session_state[state_key] = models
    st.session_state[eval_key] = eval_results
    st.session_state[split_key] = split_payload

    return models, eval_results, split_payload


# ---------------------------------------------------------
# Sidebar Configuration & Navigation
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
        <span style="font-size: 20px; font-weight: 700; color: #1D3557;">🛡️ DefectGuard AI</span><br>
        <span style="font-size: 12px; color: #64748B;">Software Defect Analytics Platform</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Dataset Selector
st.sidebar.markdown("### 📂 Dataset Selection")
dataset_choice = st.sidebar.radio(
    "Active Software Project:",
    options=["CM1 Dataset", "JM1 Dataset"],
    index=0,
    help="Select between NASA CM1 (498 modules) and JM1 (10,885 modules)."
)
selected_dataset_name = "cm1" if "CM1" in dataset_choice else "jm1"

# Load Clean Dataset
clean_df, summary, load_err = get_cached_cleaned_data(selected_dataset_name)

if load_err or clean_df is None:
    st.error(f"⚠️ Error loading dataset: {load_err}")
    st.info("Please verify that `cm1.csv` and `jm1.csv` are placed inside the `data/` directory.")
    st.stop()

# Sidebar Dataset Quick Stats
st.sidebar.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin-top: 8px; font-size: 12px;">
        <b>Active Dataset:</b> <span class="badge-neutral">{selected_dataset_name.upper()}</span><br>
        <b>Total Modules:</b> {len(clean_df):,}<br>
        <b>Defect Rate:</b> <span style="color: #E63946; font-weight: 600;">{summary.defect_rate_percent}%</span><br>
        <b>Metrics Tracked:</b> {len(summary.feature_columns)}
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Navigation Menu
st.sidebar.markdown("### 🧭 Dashboard Navigation")
navigation_pages = [
    "Overview",
    "Dataset Explorer",
    "Exploratory Data Analysis",
    "Defect Prediction",
    "Model Performance",
    "Feature Analysis",
    "Insights & Recommendations"
]
active_page = st.sidebar.radio("Navigate to:", navigation_pages, index=0)

st.sidebar.markdown("---")

# Model Training Parameters Expander in Sidebar
with st.sidebar.expander("⚙️ Model Training Controls", expanded=False):
    st.markdown("**Train/Test Partitioning**")
    test_ratio = st.slider(
        "Test Split Ratio (%):",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data held out strictly for model evaluation."
    ) / 100.0

    st.markdown("**Decision Tree Hyperparameters**")
    dt_max_depth = st.slider("Max Tree Depth:", min_value=2, max_value=15, value=5)
    dt_min_leaf = st.slider("Min Samples per Leaf:", min_value=1, max_value=20, value=4)
    dt_class_weight = st.selectbox("DT Class Weight:", ["balanced", None], index=0)

    st.markdown("**Support Vector Machine (SVM)**")
    svm_kernel = st.selectbox("SVM Kernel:", ["rbf", "linear"], index=0)
    svm_c = st.select_slider("Regularization C:", options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
    svm_class_weight = st.selectbox("SVM Class Weight:", ["balanced", None], index=0)

    retrain_clicked = st.button("🔄 Retrain Models Now", use_container_width=True)

# Train or load models
with st.spinner("Initializing models..."):
    dt_params = {
        "max_depth": dt_max_depth,
        "min_samples_leaf": dt_min_leaf,
        "class_weight": dt_class_weight
    }
    svm_params = {
        "kernel": svm_kernel,
        "c_param": svm_c,
        "class_weight": svm_class_weight
    }
    models, eval_results, split_data = ensure_models_trained(
        clean_df=clean_df,
        summary=summary,
        dataset_name=selected_dataset_name,
        test_size=test_ratio,
        force_retrain=retrain_clicked,
        dt_params=dt_params,
        svm_params=svm_params
    )

if retrain_clicked:
    st.sidebar.success("✅ Models retrained and saved successfully!")


# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <h1>Software Defect Prediction & Quality Analytics</h1>
        <p>Interactive Machine Learning Platform &bull; Active Dataset: <b>{selected_dataset_name.upper()} (NASA MDP)</b> &bull; Section: <b>{active_page}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if active_page == "Overview":
    st.subheader("Executive Project Summary")
    st.markdown(
        "Welcome to the **Software Defect Prediction System**. This dashboard applies machine learning "
        "(Decision Tree & Support Vector Machine) to static software engineering metrics "
        "to proactively detect defective software components prior to deployment."
    )

    # Best Model F1 Score calculation
    best_f1 = 0.0
    best_model_name = "N/A"
    for m_name, res in eval_results.items():
        if res["f1_score"] > best_f1:
            best_f1 = res["f1_score"]
            best_model_name = m_name

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Modules</div>
                <div class="metric-value">{len(clean_df):,}</div>
                <div class="metric-subtitle">Evaluated source routines</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        def_count = summary.class_counts.get("Defective (True)", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Defective Modules</div>
                <div class="metric-value" style="color: #E63946;">{def_count:,}</div>
                <div class="metric-subtitle">{summary.defect_rate_percent}% defect density</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        clean_count = summary.class_counts.get("Defect-Free (False)", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Defect-Free Modules</div>
                <div class="metric-value" style="color: #2A9D8F;">{clean_count:,}</div>
                <div class="metric-subtitle">{round(100 - summary.defect_rate_percent, 1)}% verified clean</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Software Metrics</div>
                <div class="metric-value">{len(summary.feature_columns)}</div>
                <div class="metric-subtitle">McCabe & Halstead features</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Best Model F1-Score</div>
                <div class="metric-value">{best_f1:.3f}</div>
                <div class="metric-subtitle">{best_model_name} (Test Split)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Overview Visualizations
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.plotly_chart(
            plot_defect_distribution(clean_df, selected_dataset_name),
            use_container_width=True
        )

    with col_chart2:
        # Cross-dataset comparison chart
        other_name = "jm1" if selected_dataset_name == "cm1" else "cm1"
        other_raw, _ = load_raw_dataset(other_name)
        if other_raw is not None:
            other_clean, _ = clean_software_dataset(other_raw, other_name)
            cm1_ref = clean_df if selected_dataset_name == "cm1" else other_clean
            jm1_ref = other_clean if selected_dataset_name == "cm1" else clean_df
            st.plotly_chart(
                plot_dataset_comparison_chart(cm1_ref, jm1_ref),
                use_container_width=True
            )
        else:
            st.info("Second dataset not available for side-by-side scale comparison.")

    # Data Quality & Preprocessing Overview
    st.markdown("### 📋 Data Quality & Hygiene Summary")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.markdown(f"**Original Dataset Dimensions:** `{summary.original_shape[0]:,} rows × {summary.original_shape[1]} columns`")
        st.markdown(f"**Cleaned Dimensions:** `{summary.cleaned_shape[0]:,} rows × {summary.cleaned_shape[1]} columns`")
    with q_col2:
        st.markdown(f"**Missing Values Identified:** `{summary.missing_values_original}` (median imputed inside pipeline)")
        st.markdown(f"**String Columns Sanitized:** `{len(summary.converted_columns)}` {summary.converted_columns if summary.converted_columns else '(None)'}")
    with q_col3:
        st.markdown(f"**Duplicate Identical Profiles:** `{summary.duplicate_metric_profiles:,}` distinct functions with shared metrics")
        st.markdown("**Data Leakage Protection:** `Fitted strictly on training folds`")

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Analytical Integrity & Scientific Disclaimer:</b><br>
            Descriptive statistics and correlations reflect historical empirical associations in the NASA Metrics Data Program (MDP). 
            Correlation does not imply direct causation: high cyclomatic complexity or lines of code correlate with increased defect likelihood, 
            but do not mechanically cause errors. Machine learning predictions are probabilistic risk estimates designed for testing triage.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2: DATASET EXPLORER
# =========================================================
elif active_page == "Dataset Explorer":
    st.subheader(f"Interactive Dataset Explorer: {selected_dataset_name.upper()}")
    st.markdown(
        "Search, inspect, filter, and export the software metrics dataset. "
        "Use the sidebar or filters below to dynamically subset modules by status and metric ranges."
    )

    # Filter Controls
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        status_filter = st.selectbox(
            "Filter by Defect Status:",
            options=["All Modules", "Defective Only (True)", "Defect-Free Only (False)"]
        )
    with exp_col2:
        loc_max_val = float(clean_df["loc"].max())
        loc_filter = st.slider(
            "Max Lines of Code (loc):",
            min_value=1.0,
            max_value=loc_max_val,
            value=loc_max_val
        )
    with exp_col3:
        selected_cols = st.multiselect(
            "Display Columns:",
            options=list(clean_df.columns),
            default=["id", "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "e", "branchCount", "defects"]
        )

    # Apply Filters
    filtered_df = clean_df.copy()
    if status_filter == "Defective Only (True)":
        filtered_df = filtered_df[filtered_df["defects"] == True]
    elif status_filter == "Defect-Free Only (False)":
        filtered_df = filtered_df[filtered_df["defects"] == False]
    
    filtered_df = filtered_df[filtered_df["loc"] <= loc_filter]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(clean_df):,} modules** based on current filters.")

    # Data Tabs
    tab_data, tab_stats, tab_meta, tab_quality = st.tabs([
        "📄 Data Table", "📊 Descriptive Statistics", "📚 Metric Dictionary", "🔍 Data Quality"
    ])

    with tab_data:
        display_subset = filtered_df[selected_cols] if selected_cols else filtered_df
        st.dataframe(display_subset, use_container_width=True, height=420)
        
        # Download Filtered CSV
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=csv_bytes,
            file_name=f"{selected_dataset_name}_filtered.csv",
            mime="text/csv",
            help="Download the filtered records without affecting the source data on disk."
        )

    with tab_stats:
        st.markdown("**Descriptive Statistical Summary (Numerical Metrics)**")
        metric_only_df = filtered_df[[c for c in filtered_df.columns if c != "id" and c in summary.feature_columns]]
        st.dataframe(metric_only_df.describe().T.round(2), use_container_width=True)

    with tab_meta:
        st.markdown("**NASA Software Metrics Glossary & Mathematical Formulations**")
        meta_rows = []
        for col, info in METRIC_DEFINITIONS.items():
            meta_rows.append({
                "Metric": col,
                "Standard Name": info.get("name"),
                "Family": info.get("family"),
                "Description": info.get("description")
            })
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

    with tab_quality:
        st.markdown("### Data Quality & Schema Inspection")
        q_cols = st.columns(2)
        with q_cols[0]:
            st.markdown("**Missing Value Breakdown:**")
            if summary.missing_values_by_column:
                st.table(pd.DataFrame(
                    list(summary.missing_values_by_column.items()),
                    columns=["Metric Column", "Missing Token Count"]
                ))
            else:
                st.success("No missing values detected in the active dataset.")
        with q_cols[1]:
            st.markdown("**Data Sanitization Log:**")
            if summary.notes:
                for note in summary.notes:
                    st.info(f"&bull; {note}")
            else:
                st.success("All columns conformed directly to standard numerical formats.")


# =========================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
elif active_page == "Exploratory Data Analysis":
    st.subheader(f"Exploratory Data Analysis: {selected_dataset_name.upper()}")
    st.markdown(
        "Analyze statistical distributions, bivariate relationships, and empirical defect rates "
        "across software complexity and sizing metrics."
    )

    # Imputed dataset for complete plotting without NaN interruptions
    eda_df = get_imputed_dataframe(clean_df, summary.feature_columns)

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "📊 Distributions & Box Plots",
        "🔗 Bivariate Metric Relationships",
        "🔥 Correlation Heatmap",
        "📈 Empirical Defect Rate Curves",
        "📋 Defective vs Clean Summary"
    ])

    with eda_tab1:
        st.markdown("#### Univariate Feature Distributions & Box Plots")
        c1, c2 = st.columns([2, 1])
        with c1:
            metric_for_dist = st.selectbox(
                "Select Software Metric for Distribution Analysis:",
                options=summary.feature_columns,
                index=0,
                format_func=lambda x: f"{x} - {METRIC_DEFINITIONS.get(x, {}).get('name', '')}"
            )
        with c2:
            use_log = st.checkbox("Apply Log Scale (recommended for skewed metrics like effort/volume)", value=False)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.plotly_chart(
                plot_feature_distribution(
                    eda_df,
                    metric_for_dist,
                    METRIC_DEFINITIONS.get(metric_for_dist, {}).get("name", metric_for_dist),
                    use_log_scale=use_log
                ),
                use_container_width=True
            )
        with col_d2:
            st.plotly_chart(
                plot_box_comparison(
                    eda_df,
                    metric_for_dist,
                    METRIC_DEFINITIONS.get(metric_for_dist, {}).get("name", metric_for_dist),
                    use_log_scale=use_log
                ),
                use_container_width=True
            )

    with eda_tab2:
        st.markdown("#### Bivariate Metric Relationships (Scatter Analysis)")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            x_metric = st.selectbox(
                "Select X-Axis Metric:",
                options=summary.feature_columns,
                index=0,
                key="scatter_x"
            )
        with sc_col2:
            y_metric = st.selectbox(
                "Select Y-Axis Metric:",
                options=summary.feature_columns,
                index=1 if len(summary.feature_columns) > 1 else 0,
                key="scatter_y"
            )

        st.plotly_chart(
            plot_scatter_relationship(eda_df, x_metric, y_metric, max_sample_points=2500),
            use_container_width=True
        )

    with eda_tab3:
        st.markdown("#### Software Metrics Correlation Matrix")
        include_target_in_corr = st.checkbox("Include 'defects' target in correlation matrix", value=True)
        st.plotly_chart(
            plot_correlation_heatmap(eda_df, summary.feature_columns, include_target=include_target_in_corr),
            use_container_width=True
        )

    with eda_tab4:
        st.markdown("#### Empirical Defect Rate Across Metric Tiers")
        st.markdown(
            "Binned risk analysis showing whether modules with higher metric values exhibit higher empirical defect percentages."
        )
        rate_metric = st.selectbox(
            "Select Metric to Evaluate Risk Tiers:",
            options=["loc", "v(g)", "branchCount", "e", "v", "d", "lOCode"],
            index=0
        )
        bins_count = st.slider("Number of Quantile Bins:", min_value=3, max_value=8, value=5)
        
        fig_rate, rate_table = plot_defect_rate_by_bins(eda_df, rate_metric, num_bins=bins_count)
        if fig_rate is not None:
            st.plotly_chart(fig_rate, use_container_width=True)
            if rate_table is not None:
                st.markdown("**Quantile Bin Empirical Statistics:**")
                st.dataframe(rate_table, use_container_width=True)
        else:
            st.warning("Insufficient distinct values to segment this metric into the selected number of bins.")

    with eda_tab5:
        st.markdown("#### Comparative Statistical Summary (Defective vs. Defect-Free)")
        group_stats_df = compute_group_statistics(eda_df, summary.feature_columns)
        st.dataframe(group_stats_df, use_container_width=True, height=450)


# =========================================================
# PAGE 4: DEFECT PREDICTION (INTERACTIVE INFERENCE)
# =========================================================
elif active_page == "Defect Prediction":
    st.subheader("Interactive Single-Module Defect Prediction")
    st.markdown(
        "Enter or adjust software engineering metrics to evaluate whether a code module is likely to contain defects. "
        "Choose between the trained **Decision Tree** and **Support Vector Machine (SVM)** classifiers."
    )

    pred_m_col1, pred_m_col2 = st.columns([1, 1])
    with pred_m_col1:
        chosen_model_name = st.selectbox(
            "Select Classifier for Inference:",
            options=["Decision Tree", "Support Vector Machine"]
        )
    with pred_m_col2:
        preset_choice = st.selectbox(
            "Load Sample Metric Preset:",
            options=[
                "Median Typical Module",
                "High-Complexity Monolithic Module (High Risk)",
                "Compact Defect-Free Stub (Low Risk)"
            ]
        )

    # Establish baseline defaults based on dataset medians
    feature_cols = summary.feature_columns
    imputed_df = get_imputed_dataframe(clean_df, feature_cols)
    defaults = {f: float(imputed_df[f].median()) for f in feature_cols}

    if preset_choice == "High-Complexity Monolithic Module (High Risk)":
        defaults["loc"] = 180.0
        defaults["v(g)"] = 28.0
        defaults["ev(g)"] = 12.0
        defaults["iv(g)"] = 16.0
        defaults["branchCount"] = 45.0
        defaults["e"] = 45000.0
        defaults["v"] = 4800.0
        defaults["d"] = 32.0
    elif preset_choice == "Compact Defect-Free Stub (Low Risk)":
        defaults["loc"] = 12.0
        defaults["v(g)"] = 2.0
        defaults["ev(g)"] = 1.0
        defaults["iv(g)"] = 1.0
        defaults["branchCount"] = 3.0
        defaults["e"] = 450.0
        defaults["v"] = 200.0
        defaults["d"] = 4.5

    st.markdown("#### Input Module Software Metrics")

    # Form with grouped metrics
    user_inputs = {}
    with st.form("single_prediction_form"):
        st.markdown("**1. Sizing & Line Count Metrics**")
        g1_c1, g1_c2, g1_c3, g1_c4, g1_c5 = st.columns(5)
        with g1_c1:
            user_inputs["loc"] = st.number_input("LOC (Lines of Code)", value=float(defaults.get("loc", 25.0)), min_value=0.0)
        with g1_c2:
            user_inputs["lOCode"] = st.number_input("lOCode (Executable)", value=float(defaults.get("lOCode", 18.0)), min_value=0.0)
        with g1_c3:
            user_inputs["lOComment"] = st.number_input("lOComment (Comments)", value=float(defaults.get("lOComment", 2.0)), min_value=0.0)
        with g1_c4:
            user_inputs["lOBlank"] = st.number_input("lOBlank (Blank)", value=float(defaults.get("lOBlank", 4.0)), min_value=0.0)
        with g1_c5:
            user_inputs["locCodeAndComment"] = st.number_input("locCodeAndComment", value=float(defaults.get("locCodeAndComment", 0.0)), min_value=0.0)

        st.markdown("**2. McCabe Complexity & Branch Metrics**")
        g2_c1, g2_c2, g2_c3, g2_c4 = st.columns(4)
        with g2_c1:
            user_inputs["v(g)"] = st.number_input("v(g) (Cyclomatic)", value=float(defaults.get("v(g)", 3.0)), min_value=0.0)
        with g2_c2:
            user_inputs["ev(g)"] = st.number_input("ev(g) (Essential)", value=float(defaults.get("ev(g)", 1.0)), min_value=0.0)
        with g2_c3:
            user_inputs["iv(g)"] = st.number_input("iv(g) (Design)", value=float(defaults.get("iv(g)", 2.0)), min_value=0.0)
        with g2_c4:
            user_inputs["branchCount"] = st.number_input("branchCount (Branches)", value=float(defaults.get("branchCount", 5.0)), min_value=0.0)

        st.markdown("**3. Halstead Operator & Operand Counts**")
        g3_c1, g3_c2, g3_c3, g3_c4, g3_c5 = st.columns(5)
        with g3_c1:
            user_inputs["uniq_Op"] = st.number_input("uniq_Op (Unique Op)", value=float(defaults.get("uniq_Op", 10.0)), min_value=0.0)
        with g3_c2:
            user_inputs["uniq_Opnd"] = st.number_input("uniq_Opnd (Unique Opnd)", value=float(defaults.get("uniq_Opnd", 14.0)), min_value=0.0)
        with g3_c3:
            user_inputs["total_Op"] = st.number_input("total_Op (Total Op)", value=float(defaults.get("total_Op", 45.0)), min_value=0.0)
        with g3_c4:
            user_inputs["total_Opnd"] = st.number_input("total_Opnd (Total Opnd)", value=float(defaults.get("total_Opnd", 30.0)), min_value=0.0)
        with g3_c5:
            user_inputs["n"] = st.number_input("n (Halstead Total N)", value=float(defaults.get("n", 75.0)), min_value=0.0)

        st.markdown("**4. Halstead Derived Metrics**")
        g4_c1, g4_c2, g4_c3, g4_c4, g4_c5, g4_c6, g4_c7 = st.columns(7)
        with g4_c1:
            user_inputs["v"] = st.number_input("v (Volume)", value=float(defaults.get("v", 380.0)), min_value=0.0)
        with g4_c2:
            user_inputs["l"] = st.number_input("l (Level)", value=float(defaults.get("l", 0.08)), min_value=0.0)
        with g4_c3:
            user_inputs["d"] = st.number_input("d (Difficulty)", value=float(defaults.get("d", 12.0)), min_value=0.0)
        with g4_c4:
            user_inputs["i"] = st.number_input("i (Intelligence)", value=float(defaults.get("i", 28.0)), min_value=0.0)
        with g4_c5:
            user_inputs["e"] = st.number_input("e (Effort)", value=float(defaults.get("e", 4500.0)), min_value=0.0)
        with g4_c6:
            user_inputs["b"] = st.number_input("b (Errors B)", value=float(defaults.get("b", 0.12)), min_value=0.0)
        with g4_c7:
            user_inputs["t"] = st.number_input("t (Time T sec)", value=float(defaults.get("t", 250.0)), min_value=0.0)

        submit_prediction = st.form_submit_button("⚡ Predict Defect Likelihood", use_container_width=True)

    if submit_prediction:
        selected_pipeline = models.get(chosen_model_name)
        is_valid, validation_msgs = validate_module_inputs(user_inputs, feature_cols)

        if not is_valid:
            for m in validation_msgs:
                st.error(m)
        else:
            result = predict_defect_risk(selected_pipeline, user_inputs, feature_cols)

            st.markdown("### Prediction Results & Risk Assessment")
            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                if result["is_defective"]:
                    st.markdown(
                        f"""
                        <div style="background-color: #FEF2F2; border: 2px solid #E63946; border-radius: 8px; padding: 20px; text-align: center;">
                            <span style="font-size: 32px;">⚠️</span><br>
                            <span style="font-size: 20px; font-weight: 700; color: #991B1B;">PREDICTED STATUS: DEFECTIVE</span><br>
                            <span style="font-size: 14px; color: #7F1D1D; margin-top: 6px; display: inline-block;">
                                High probability of post-release defects or testing failures.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #F0FDF4; border: 2px solid #2A9D8F; border-radius: 8px; padding: 20px; text-align: center;">
                            <span style="font-size: 32px;">✅</span><br>
                            <span style="font-size: 20px; font-weight: 700; color: #166534;">PREDICTED STATUS: DEFECT-FREE</span><br>
                            <span style="font-size: 14px; color: #14532D; margin-top: 6px; display: inline-block;">
                                Module metrics fall within historical defect-free distributions.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with res_c2:
                if result["has_probability"]:
                    st.markdown(f"**Defect Likelihood Score:** `{result['risk_percentage']}%`")
                    st.progress(min(1.0, max(0.0, result["risk_percentage"] / 100.0)))
                    st.markdown(
                        f"**Assessed Risk Category:** "
                        f"<span style='color: {result['risk_color']}; font-weight: 700;'>{result['risk_tier']}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Calibrated probability scores not supported for this configuration.")

                if result["diagnostic_flags"]:
                    st.markdown("**Diagnostic Observations:**")
                    for flag in result["diagnostic_flags"]:
                        st.warning(f"&bull; {flag}")
                else:
                    st.success("No metric values significantly breached standard complexity risk thresholds.")

            st.markdown(f"<div class='disclaimer-box'>{result['disclaimer']}</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 5: MODEL PERFORMANCE
# =========================================================
elif active_page == "Model Performance":
    st.subheader(f"Model Evaluation & Comparison: {selected_dataset_name.upper()}")
    st.markdown(
        "Evaluate Decision Tree and Support Vector Machine classifiers on held-out test data. "
        "Models are evaluated with the positive class defined as **Defective Module (1)**."
    )

    # Comparison Table
    st.markdown("### 📊 Test Set Metrics Comparison")
    comp_df = build_comparison_dataframe(eval_results)
    st.dataframe(comp_df, use_container_width=True)

    # Grouped Bar Chart
    st.plotly_chart(plot_model_comparison_bars(eval_results), use_container_width=True)

    st.markdown("---")

    # Confusion Matrices
    st.markdown("### 🎯 Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.plotly_chart(
            plot_confusion_matrix(eval_results["Decision Tree"]["confusion_matrix"], "Decision Tree", "Test"),
            use_container_width=True
        )
    with cm_col2:
        st.plotly_chart(
            plot_confusion_matrix(eval_results["Support Vector Machine"]["confusion_matrix"], "Support Vector Machine", "Test"),
            use_container_width=True
        )

    st.markdown("---")

    # ROC and PR Curves
    st.markdown("### 📈 Discriminative Ability: ROC & Precision-Recall Curves")
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.plotly_chart(plot_roc_curves(eval_results), use_container_width=True)
    with curve_col2:
        baseline_rate = summary.defect_rate_percent / 100.0
        st.plotly_chart(plot_precision_recall_curves(eval_results, baseline_defect_rate=baseline_rate), use_container_width=True)

    # Classification Reports Expander
    with st.expander("📝 Detailed Scikit-Learn Classification Reports", expanded=False):
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown("**Decision Tree Classification Report:**")
            dt_rep = eval_results["Decision Tree"]["classification_report"]
            st.json(dt_rep)
        with rep_col2:
            st.markdown("**Support Vector Machine Classification Report:**")
            svm_rep = eval_results["Support Vector Machine"]["classification_report"]
            st.json(svm_rep)

    # Metric Guide Box
    st.markdown("### 💡 Plain-Language Metric Guide for QA Engineers")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(f"**Recall (Sensitivity):** {METRIC_EXPLANATIONS['recall']}")
        st.markdown(f"**Precision:** {METRIC_EXPLANATIONS['precision']}")
        st.markdown(f"**F1-Score:** {METRIC_EXPLANATIONS['f1_score']}")
    with g_col2:
        st.markdown(f"**ROC-AUC:** {METRIC_EXPLANATIONS['roc_auc']}")
        st.markdown(f"**Average Precision:** {METRIC_EXPLANATIONS['avg_precision']}")
        st.markdown(f"**Accuracy:** {METRIC_EXPLANATIONS['accuracy']}")


# =========================================================
# PAGE 6: FEATURE ANALYSIS
# =========================================================
elif active_page == "Feature Analysis":
    st.subheader(f"Feature Importance & Predictive Associations: {selected_dataset_name.upper()}")
    st.markdown(
        "Analyze which software engineering metrics carry the strongest predictive weight in identifying defective modules. "
        "Gini impurity reduction is calculated for the Decision Tree, and Permutation Importance on held-out test data is calculated for SVM."
    )

    top_n_features = st.slider("Number of Top Features to Display:", min_value=5, max_value=21, value=10)

    # Decision Tree Importance
    dt_pipeline = models["Decision Tree"]
    dt_importance_df = get_decision_tree_importance(dt_pipeline, summary.feature_columns)

    # SVM Permutation Importance
    X_test_data = split_data["X_test"]
    y_test_data = split_data["y_test"]
    svm_pipeline = models["Support Vector Machine"]

    with st.spinner("Computing permutation importance on held-out test split..."):
        svm_importance_df = get_svm_permutation_importance(
            svm_pipeline,
            X_test_data,
            y_test_data,
            summary.feature_columns,
            n_repeats=3
        )

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.plotly_chart(
            plot_feature_importance_bar(
                dt_importance_df,
                title="Decision Tree: Gini Feature Importance",
                top_n=top_n_features,
                metric_col="Importance"
            ),
            use_container_width=True
        )
    with feat_col2:
        st.plotly_chart(
            plot_feature_importance_bar(
                svm_importance_df,
                title="SVM: Permutation Importance (Test Set)",
                top_n=top_n_features,
                metric_col="Importance_Mean"
            ),
            use_container_width=True
        )

    st.markdown("### 📑 Detailed Feature Importance Rankings")
    tab_dt_table, tab_svm_table = st.tabs(["Decision Tree Importance Table", "SVM Permutation Importance Table"])
    with tab_dt_table:
        st.dataframe(dt_importance_df.head(top_n_features), use_container_width=True)
    with tab_svm_table:
        st.dataframe(svm_importance_df.head(top_n_features), use_container_width=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Interpretation Note on Feature Importance vs. Causality:</b><br>
            • <b>Decision Tree Gini Importance:</b> Measures the total normalized reduction in Gini impurity brought by that feature across all splits.<br>
            • <b>SVM Permutation Importance:</b> Measures the drop in test set F1-score when that feature's values are randomly shuffled. (Ordinary linear SVM coefficients do not apply directly to RBF non-linear kernels).<br>
            • <b>Causality Disclaimer:</b> Having high feature importance means a metric has strong statistical predictive power in classifying defects. It does <i>not</i> prove that the metric directly caused the defect.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 7: INSIGHTS & RECOMMENDATIONS
# =========================================================
elif active_page == "Insights & Recommendations":
    st.subheader("Actionable Quality Assurance Insights & Project Limitations")
    st.markdown(
        "Synthesized recommendations derived from the actual dataset distributions and trained machine learning models, "
        "accompanied by academic defense limitations for thesis reports."
    )

    qa_recs = generate_qa_recommendations(
        dataset_name=selected_dataset_name,
        df=clean_df,
        eval_results=eval_results,
        top_features=summary.feature_columns
    )

    st.markdown("### 🎯 Quality Assurance & Engineering Recommendations")
    for rec in qa_recs:
        p_class = "high-priority" if rec["priority"] == "Critical" or rec["priority"] == "High" else (
            "medium-priority" if rec["priority"] == "Medium" else "low-priority"
        )
        badge_color = "#E63946" if "High" in rec["priority"] or "Critical" in rec["priority"] else (
            "#F4A261" if "Medium" in rec["priority"] else "#2A9D8F"
        )

        st.markdown(
            f"""
            <div class="rec-card {p_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 16px; color: #1D3557;">{rec['title']}</span>
                    <span style="background-color: {badge_color}; color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                        {rec['priority']} Priority
                    </span>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-bottom: 8px;"><b>Category:</b> {rec['category']}</div>
                <div style="font-size: 13px; color: #334155; margin-bottom: 6px;"><b>Observed Evidence:</b> {rec['observation']}</div>
                <div style="font-size: 13px; color: #1E293B;"><b>Recommended Engineering Action:</b> {rec['action']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### ⚠️ Academic Project Limitations & Defense Briefing")
    st.markdown(
        "When defending this project to evaluators or writing the project report, "
        "incorporate the following critical analytical boundaries:"
    )

    limitations = get_academic_limitations()
    for lim in limitations:
        with st.expander(f"📌 {lim['aspect']}", expanded=True):
            st.markdown(lim["description"])

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Final Project Report Guidance:</b><br>
            Always contextualize model performance within the domain of software testing. Missing a bug (False Negative) 
            can lead to severe production outages or mission failure in aerospace software, whereas a false alarm (False Positive) 
            merely costs an engineer 15 minutes of additional code review. Therefore, models tuned for higher Recall and F1-score 
            are preferred over naive high-accuracy models in real-world QA engineering.
        </div>
        """,
        unsafe_allow_html=True
    )
