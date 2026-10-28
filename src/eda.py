"""
Exploratory Data Analysis (EDA) Module for Software Defect Prediction.

Generates interactive Plotly visualizations for software metrics distributions,
class imbalance, correlations, bivariate metric relationships, and defect rates.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Professional navy, blue, neutral corporate theme colors
COLOR_DEFECTIVE = "#E63946"      # Clear coral red for defects
COLOR_DEFECT_FREE = "#1D3557"    # Deep corporate navy for defect-free
COLOR_PRIMARY = "#2A9D8F"        # Teal accent
COLOR_SECONDARY = "#457B9D"      # Steel blue
COLOR_BACKGROUND = "#FFFFFF"     # White
COLOR_GRID = "#E5E5E5"           # Light gray grid lines
COLOR_PALETTE = ["#1D3557", "#457B9D", "#2A9D8F", "#E63946", "#F4A261", "#E76F51"]


def set_plot_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies a clean, modern aesthetic with readable typography and subtle gridlines."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": "#1D3557", "family": "Inter, sans-serif"}
        },
        template="plotly_white",
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E5E5E5",
            borderwidth=1
        ),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12, color="#333333"),
        hoverlabel=dict(bgcolor="#1D3557", font_size=12, font_color="#FFFFFF")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLOR_GRID, zeroline=False)
    return fig


def plot_defect_distribution(df: pd.DataFrame, dataset_name: str = "Dataset") -> go.Figure:
    """
    Creates an interactive bar chart and donut chart of defective vs defect-free modules.
    """
    counts = df["defects"].value_counts()
    false_count = int(counts.get(False, 0))
    true_count = int(counts.get(True, 0))
    total = false_count + true_count

    labels = ["Defect-Free (False)", "Defective (True)"]
    vals = [false_count, true_count]
    pcts = [round(false_count / total * 100, 2), round(true_count / total * 100, 2)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Module Count by Status", "Defect Proportion"),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=vals,
            text=[f"{v:,} ({p}%)" for v, p in zip(vals, pcts)],
            textposition="auto",
            marker_color=[COLOR_DEFECT_FREE, COLOR_DEFECTIVE],
            hoverinfo="x+y+text",
            showlegend=False
        ),
        row=1, col=1
    )

    # Donut chart
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=vals,
            hole=0.55,
            marker_colors=[COLOR_DEFECT_FREE, COLOR_DEFECTIVE],
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            showlegend=False
        ),
        row=1, col=2
    )

    title = f"{dataset_name.upper()} - Class Distribution (Total Modules: {total:,})"
    set_plot_theme(fig, title)
    fig.update_layout(height=380)
    return fig


def plot_feature_distribution(
    df: pd.DataFrame,
    feature_name: str,
    metric_label: str = "",
    use_log_scale: bool = False
) -> go.Figure:
    """
    Interactive histogram and KDE-like distribution for a selected metric, grouped by defect status.
    """
    display_title = metric_label if metric_label else feature_name
    plot_df = df[[feature_name, "defects"]].dropna().copy()
    plot_df["Status"] = plot_df["defects"].map({True: "Defective", False: "Defect-Free"})

    fig = px.histogram(
        plot_df,
        x=feature_name,
        color="Status",
        color_discrete_map={"Defective": COLOR_DEFECTIVE, "Defect-Free": COLOR_DEFECT_FREE},
        barmode="overlay",
        marginal="box",
        opacity=0.7,
        log_x=use_log_scale,
        labels={feature_name: display_title, "Status": "Module Status"},
        title=f"Distribution of {display_title}"
    )

    set_plot_theme(fig, f"Distribution of {display_title}")
    fig.update_layout(height=450, yaxis_title="Number of Modules")
    return fig


def plot_box_comparison(
    df: pd.DataFrame,
    feature_name: str,
    metric_label: str = "",
    use_log_scale: bool = False
) -> go.Figure:
    """
    Box plot comparing metric values between Defective and Defect-Free modules.
    """
    display_title = metric_label if metric_label else feature_name
    plot_df = df[[feature_name, "defects"]].dropna().copy()
    plot_df["Status"] = plot_df["defects"].map({True: "Defective", False: "Defect-Free"})

    fig = px.box(
        plot_df,
        x="Status",
        y=feature_name,
        color="Status",
        color_discrete_map={"Defective": COLOR_DEFECTIVE, "Defect-Free": COLOR_DEFECT_FREE},
        points="outliers",
        log_y=use_log_scale,
        labels={feature_name: display_title, "Status": "Module Status"},
        title=f"{display_title} by Defect Status"
    )

    set_plot_theme(fig, f"{display_title} Comparison (Defective vs Defect-Free)")
    fig.update_layout(height=420, showlegend=False)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    include_target: bool = True
) -> go.Figure:
    """
    Calculates Pearson correlation matrix among software metrics (and optionally defect target).
    Excludes identifier 'id'.
    """
    valid_cols = [c for c in features if c in df.columns and c != "id"]
    if include_target and "defects" in df.columns:
        valid_cols = valid_cols + ["defects"]

    # Compute correlation
    corr_df = df[valid_cols].astype(float).corr()

    # Mask constant/NaN columns if any
    corr_matrix = corr_df.fillna(0.0).round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
        title="Software Metrics Correlation Matrix"
    )

    set_plot_theme(fig, "Correlation Matrix (McCabe & Halstead Metrics)")
    fig.update_layout(height=650, coloraxis_colorbar=dict(title="Pearson r"))
    return fig


def plot_scatter_relationship(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    max_sample_points: int = 3000
) -> go.Figure:
    """
    Interactive scatter plot comparing two software metrics with defect status overlay.
    Performs deterministic sampling if dataset is large (e.g. JM1 with 10k+ rows) to preserve browser responsiveness.
    """
    plot_df = df[[x_metric, y_metric, "defects"]].dropna().copy()
    total_points = len(plot_df)

    if total_points > max_sample_points:
        # Stratified sampling to preserve minority defective points
        defective = plot_df[plot_df["defects"] == True]
        defect_free = plot_df[plot_df["defects"] == False]

        def_ratio = len(defective) / total_points
        n_def = int(max_sample_points * def_ratio)
        n_nondef = max_sample_points - n_def

        sample_def = defective.sample(n=min(len(defective), n_def), random_state=42)
        sample_nondef = defect_free.sample(n=min(len(defect_free), n_nondef), random_state=42)
        sampled = pd.concat([sample_def, sample_nondef])
        subtitle = f" (Displaying representative sample of {len(sampled):,} / {total_points:,} modules)"
    else:
        sampled = plot_df
        subtitle = f" ({total_points:,} modules)"

    sampled["Status"] = sampled["defects"].map({True: "Defective", False: "Defect-Free"})

    fig = px.scatter(
        sampled,
        x=x_metric,
        y=y_metric,
        color="Status",
        color_discrete_map={"Defective": COLOR_DEFECTIVE, "Defect-Free": COLOR_DEFECT_FREE},
        opacity=0.65,
        hover_data={"Status": True, x_metric: ":.2f", y_metric: ":.2f"},
        labels={"Status": "Defect Status"}
    )

    set_plot_theme(fig, f"{x_metric} vs. {y_metric}{subtitle}")
    fig.update_layout(height=450)
    return fig


def plot_defect_rate_by_bins(
    df: pd.DataFrame,
    metric: str,
    num_bins: int = 5
) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """
    Segments a software metric into quantile bins and calculates the empirical defect rate in each bin.
    Provides sample sizes to prevent misleading conclusions.
    """
    clean_series = df[[metric, "defects"]].dropna()
    if clean_series.empty or clean_series[metric].nunique() < num_bins:
        return None, None

    try:
        clean_series["bin"] = pd.qcut(clean_series[metric], q=num_bins, duplicates="drop")
    except Exception:
        clean_series["bin"] = pd.cut(clean_series[metric], bins=num_bins)

    stats = clean_series.groupby("bin", observed=False).agg(
        total_modules=("defects", "count"),
        defective_modules=("defects", "sum"),
        defect_rate=("defects", lambda x: round(float(x.mean() * 100), 2))
    ).reset_index()

    stats["bin_label"] = stats["bin"].astype(str)

    fig = go.Figure()
    # Bar for total modules (secondary axis)
    fig.add_trace(
        go.Bar(
            x=stats["bin_label"],
            y=stats["total_modules"],
            name="Sample Size (Modules)",
            marker_color="#CBD5E1",
            yaxis="y1"
        )
    )
    # Line for defect rate (primary axis)
    fig.add_trace(
        go.Scatter(
            x=stats["bin_label"],
            y=stats["defect_rate"],
            name="Defect Rate (%)",
            mode="lines+markers+text",
            text=[f"{r}%" for r in stats["defect_rate"]],
            textposition="top center",
            marker=dict(size=9, color=COLOR_DEFECTIVE),
            line=dict(color=COLOR_DEFECTIVE, width=3),
            yaxis="y2"
        )
    )

    fig.update_layout(
        yaxis=dict(title="Sample Size (Modules)", side="left", showgrid=True),
        yaxis2=dict(
            title="Defect Rate (%)",
            side="right",
            overlaying="y",
            showgrid=False,
            range=[0, max(stats["defect_rate"].max() * 1.25, 20)]
        ),
        xaxis=dict(title=f"Metric Value Ranges ({metric})"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    set_plot_theme(fig, f"Empirical Defect Rate Across {metric} Tiers")
    return fig, stats


def compute_group_statistics(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Computes comparative statistical summary (Mean, Median, Std, Min, Max)
    stratified by Defect-Free vs Defective modules.
    """
    records = []
    for f in features:
        if f not in df.columns or f == "id" or f == "defects":
            continue
        col_all = df[f].dropna()
        col_defect = df[df["defects"] == True][f].dropna()
        col_clean = df[df["defects"] == False][f].dropna()

        records.append({
            "Metric": f,
            "Overall Median": round(col_all.median(), 2),
            "Defect-Free Median": round(col_clean.median(), 2),
            "Defective Median": round(col_defect.median(), 2),
            "Defect-Free Mean": round(col_clean.mean(), 2),
            "Defective Mean": round(col_defect.mean(), 2),
            "Ratio (Def/Clean Mean)": round(col_defect.mean() / (col_clean.mean() + 1e-6), 2)
        })
    return pd.DataFrame(records)


def plot_dataset_comparison_chart(cm1_df: pd.DataFrame, jm1_df: pd.DataFrame) -> go.Figure:
    """
    Visual comparison between CM1 and JM1 dataset scale and defect percentages.
    """
    cm1_total = len(cm1_df)
    cm1_def = int((cm1_df["defects"] == True).sum())
    cm1_rate = round(cm1_def / cm1_total * 100, 2)

    jm1_total = len(jm1_df)
    jm1_def = int((jm1_df["defects"] == True).sum())
    jm1_rate = round(jm1_def / jm1_total * 100, 2)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Dataset Module Volume (Total Size)", "Observed Defect Rate (%)")
    )

    fig.add_trace(
        go.Bar(
            x=["CM1 (NASA)", "JM1 (NASA)"],
            y=[cm1_total, jm1_total],
            text=[f"{cm1_total:,} modules", f"{jm1_total:,} modules"],
            textposition="auto",
            marker_color=[COLOR_SECONDARY, COLOR_DEFECT_FREE],
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=["CM1 (NASA)", "JM1 (NASA)"],
            y=[cm1_rate, jm1_rate],
            text=[f"{cm1_rate}%", f"{jm1_rate}%"],
            textposition="auto",
            marker_color=[COLOR_DEFECTIVE, COLOR_DEFECTIVE],
            showlegend=False
        ),
        row=1, col=2
    )

    set_plot_theme(fig, "CM1 vs. JM1 Dataset Comparison")
    fig.update_layout(height=360)
    return fig
