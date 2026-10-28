"""
Preprocessing Module for Software Defect Prediction.

Handles cleaning, numeric conversion of string columns (e.g., in JM1),
missing value detection and imputation, target binarization,
duplicate analysis, and leak-free train/test splitting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


@dataclass
class PreprocessingSummary:
    """Stores detailed statistics regarding dataset preprocessing."""
    dataset_name: str
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    missing_values_original: int
    missing_values_by_column: Dict[str, int]
    duplicate_rows_total: int
    duplicate_metric_profiles: int
    feature_columns: List[str]
    target_column: str
    class_counts: Dict[str, int]
    defect_rate_percent: float
    converted_columns: List[str]
    notes: List[str] = field(default_factory=list)


def clean_software_dataset(
    df: pd.DataFrame,
    dataset_name: str = "dataset"
) -> Tuple[pd.DataFrame, PreprocessingSummary]:
    """
    Cleans raw NASA MDP dataset:
    - Normalizes column names (strips whitespace).
    - Preserves 'id' column without modification.
    - Robustly converts metric columns to numeric (handling '?' and string representations).
    - Normalizes 'defects' target to boolean / binary (1 = defective, 0 = defect-free).
    - Generates a transparent summary of data quality.
    
    Parameters:
        df: Raw DataFrame loaded from CSV.
        dataset_name: Identifier name (e.g. 'cm1', 'jm1').
        
    Returns:
        Tuple of (Cleaned DataFrame with numeric metrics and boolean defects, PreprocessingSummary).
    """
    clean_df = df.copy()
    original_shape = clean_df.shape
    notes: List[str] = []

    # Clean whitespace in column names
    clean_df.columns = [str(c).strip() for c in clean_df.columns]

    # Target column validation
    if "defects" not in clean_df.columns:
        raise ValueError("Missing required target column 'defects' in dataset.")

    # Target normalization to clean boolean
    raw_target = clean_df["defects"]
    if raw_target.dtype == object:
        # String representations like 'true', 'false', 'Y', 'N', etc.
        target_mapping = {
            "true": True, "t": True, "yes": True, "y": True, "1": True, "defective": True,
            "false": False, "f": False, "no": False, "n": False, "0": False, "defect-free": False
        }
        clean_df["defects"] = raw_target.astype(str).str.strip().str.lower().map(target_mapping)
        if clean_df["defects"].isna().any():
            invalid_cnt = clean_df["defects"].isna().sum()
            notes.append(f"Found {invalid_cnt} unrecognized target values; coercing to False (majority class).")
            clean_df["defects"] = clean_df["defects"].fillna(False)
    else:
        clean_df["defects"] = clean_df["defects"].astype(bool)

    # Identify metric columns (exclude id and defects)
    exclude_cols = {"id", "defects"}
    metric_cols = [c for c in clean_df.columns if c not in exclude_cols]

    converted_cols: List[str] = []
    # Convert metric columns to float numeric safely
    for col in metric_cols:
        orig_dtype = clean_df[col].dtype
        if orig_dtype == object:
            # Check for non-numeric tokens such as '?' or spaces
            converted = pd.to_numeric(clean_df[col], errors="coerce")
            num_coerced = converted.isna().sum() - clean_df[col].isna().sum()
            if num_coerced > 0:
                notes.append(f"Column '{col}' had {num_coerced} non-numeric token(s) (e.g. '?') converted to NaN.")
                converted_cols.append(col)
            clean_df[col] = converted
        else:
            # Replace any infinite values with NaN
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            
        # Replace inf/-inf if any with NaN
        inf_mask = np.isinf(clean_df[col])
        if inf_mask.any():
            notes.append(f"Replaced {inf_mask.sum()} infinite value(s) in '{col}' with NaN.")
            clean_df.loc[inf_mask, col] = np.nan

    # Missing values accounting
    missing_by_col = clean_df[metric_cols].isna().sum()
    missing_by_col_dict = {col: int(cnt) for col, cnt in missing_by_col.items() if cnt > 0}
    total_missing = int(missing_by_col.sum())

    # Duplicate analysis
    total_dups = int(clean_df.duplicated().sum())  # with id
    metric_dups = int(clean_df.duplicated(subset=metric_cols).sum())  # without id
    if metric_dups > 0:
        notes.append(
            f"Found {metric_dups} module(s) sharing identical software metric profiles. "
            "These represent distinct functions/methods with their own ID, retained to avoid data loss."
        )

    # Class distribution
    true_cnt = int((clean_df["defects"] == True).sum())
    false_cnt = int((clean_df["defects"] == False).sum())
    total_records = len(clean_df)
    defect_rate = round((true_cnt / total_records * 100) if total_records > 0 else 0.0, 2)

    summary = PreprocessingSummary(
        dataset_name=dataset_name,
        original_shape=original_shape,
        cleaned_shape=clean_df.shape,
        missing_values_original=total_missing,
        missing_values_by_column=missing_by_col_dict,
        duplicate_rows_total=total_dups,
        duplicate_metric_profiles=metric_dups,
        feature_columns=metric_cols,
        target_column="defects",
        class_counts={"Defective (True)": true_cnt, "Defect-Free (False)": false_cnt},
        defect_rate_percent=defect_rate,
        converted_columns=converted_cols,
        notes=notes
    )

    return clean_df, summary


def get_imputed_dataframe(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Returns a copy of the dataframe where missing numeric values in feature_cols
    are median-imputed. Used strictly for EDA and visualization components where complete
    numeric records are required for plotting.
    (Model training uses scikit-learn Pipeline with SimpleImputer to prevent data leakage).
    """
    df_imp = df.copy()
    if df_imp[feature_cols].isna().any().any():
        imputer = SimpleImputer(strategy="median")
        df_imp[feature_cols] = imputer.fit_transform(df_imp[feature_cols])
    return df_imp


def split_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "defects",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Splits features and target into train and test sets.
    Features strictly exclude 'id' and 'defects' to prevent target leakage.
    
    Parameters:
        df: Cleaned DataFrame.
        feature_cols: List of software metric column names.
        target_col: Name of target column (default 'defects').
        test_size: Proportion for test set (default 0.20).
        random_state: Seed for reproducibility.
        stratify: Whether to use stratified sampling based on class distribution.
        
    Returns:
        X_train, X_test, y_train, y_test, split_metadata
    """
    # Guard against accidental leakage
    clean_features = [c for c in feature_cols if c not in ["id", target_col]]
    
    X = df[clean_features].copy()
    # Cast boolean target to int (0, 1) for standardized ML estimators
    y = df[target_col].astype(int).copy()

    # Determine if stratification is feasible
    stratify_target = None
    min_class_count = y.value_counts().min()
    stratified_used = False
    
    if stratify and min_class_count >= 2:
        stratify_target = y
        stratified_used = True

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size_ratio": test_size,
        "feature_count": len(clean_features),
        "feature_names": clean_features,
        "stratified": stratified_used,
        "random_state": random_state,
        "train_defective_count": int((y_train == 1).sum()),
        "train_defect_rate": round(float((y_train == 1).mean() * 100), 2),
        "test_defective_count": int((y_test == 1).sum()),
        "test_defect_rate": round(float((y_test == 1).mean() * 100), 2),
    }

    return X_train, X_test, y_train, y_test, metadata
