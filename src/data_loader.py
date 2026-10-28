"""
Data Loader Module for Software Defect Prediction.

Handles loading of NASA MDP datasets (CM1, JM1), validation of schemas,
and initial data extraction with friendly error handling.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Dictionary of metric metadata for tooltips, explanations, and visualization titles
METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "loc": {
        "name": "Lines of Code (LOC)",
        "family": "McCabe",
        "description": "Total lines of code in the software module, representing module size."
    },
    "v(g)": {
        "name": "Cyclomatic Complexity",
        "family": "McCabe",
        "description": "Number of linearly independent paths through the code; measures control flow complexity."
    },
    "ev(g)": {
        "name": "Essential Complexity",
        "family": "McCabe",
        "description": "Degree of unstructuredness in control flow; high values indicate spaghetti code."
    },
    "iv(g)": {
        "name": "Design Complexity",
        "family": "McCabe",
        "description": "Complexity of module calling graph and inter-module coupling."
    },
    "n": {
        "name": "Halstead Total Operators & Operands",
        "family": "Halstead",
        "description": "Total count of all operator tokens and operand tokens (N = N1 + N2)."
    },
    "v": {
        "name": "Halstead Volume",
        "family": "Halstead",
        "description": "Information content / size of implementation in bits (V = N * log2(unique tokens))."
    },
    "l": {
        "name": "Halstead Program Level",
        "family": "Halstead",
        "description": "Inverse of difficulty; compact, high-level code has higher level."
    },
    "d": {
        "name": "Halstead Difficulty",
        "family": "Halstead",
        "description": "Difficulty of writing or understanding the program."
    },
    "i": {
        "name": "Halstead Intelligence",
        "family": "Halstead",
        "description": "Information content independent of language choice (I = V / D)."
    },
    "e": {
        "name": "Halstead Effort",
        "family": "Halstead",
        "description": "Mental effort required to develop or comprehend the module."
    },
    "b": {
        "name": "Halstead Estimated Errors",
        "family": "Halstead",
        "description": "Calculated prediction of defects or bugs present in the module (B = V / 3000)."
    },
    "t": {
        "name": "Halstead Time to Implement",
        "family": "Halstead",
        "description": "Estimated implementation time in seconds (T = E / 18)."
    },
    "lOCode": {
        "name": "Executable Lines of Code",
        "family": "Size",
        "description": "Number of actual executable instructions excluding comments and blank lines."
    },
    "lOComment": {
        "name": "Comment Lines",
        "family": "Size",
        "description": "Number of documentation / comment lines."
    },
    "lOBlank": {
        "name": "Blank Lines",
        "family": "Size",
        "description": "Number of empty whitespace lines for readability."
    },
    "locCodeAndComment": {
        "name": "Code & Comment Lines",
        "family": "Size",
        "description": "Lines containing both executable logic and comments."
    },
    "uniq_Op": {
        "name": "Unique Operators",
        "family": "Halstead",
        "description": "Count of distinct operator symbols (mu1)."
    },
    "uniq_Opnd": {
        "name": "Unique Operands",
        "family": "Halstead",
        "description": "Count of distinct variable and constant tokens (mu2)."
    },
    "total_Op": {
        "name": "Total Operators",
        "family": "Halstead",
        "description": "Total count of all operator occurrences (N1)."
    },
    "total_Opnd": {
        "name": "Total Operands",
        "family": "Halstead",
        "description": "Total count of all operand occurrences (N2)."
    },
    "branchCount": {
        "name": "Branch Count",
        "family": "McCabe",
        "description": "Total number of decision branch exits in the module flow graph."
    },
    "defects": {
        "name": "Defect Status",
        "family": "Target",
        "description": "Binary ground-truth label: True if module contained reported defects, False if defect-free."
    }
}


def get_default_data_dir() -> Path:
    """Returns the absolute Path to the project's data directory."""
    # Anchored relative to the project root (parent of src)
    return Path(__file__).resolve().parent.parent / "data"


def load_raw_dataset(dataset_name: str, data_dir: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads raw CSV dataset (either cm1 or jm1) from the data directory.
    
    Parameters:
        dataset_name: 'cm1' or 'jm1' (case-insensitive).
        data_dir: Optional custom Path to the directory containing CSV files.
        
    Returns:
        Tuple of (DataFrame or None, error_message or None).
    """
    clean_name = dataset_name.strip().lower()
    if clean_name not in ["cm1", "jm1"]:
        return None, f"Unsupported dataset name '{dataset_name}'. Please select 'cm1' or 'jm1'."

    target_dir = Path(data_dir) if data_dir else get_default_data_dir()
    file_path = target_dir / f"{clean_name}.csv"

    if not file_path.exists():
        return None, (
            f"Dataset file not found at '{file_path}'. "
            f"Please ensure '{clean_name}.csv' is placed inside the 'data/' folder."
        )

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None, f"Dataset file '{file_path.name}' is empty."
        return df, None
    except Exception as e:
        return None, f"Failed to read CSV file '{file_path.name}': {str(e)}"


def get_dataset_inventory(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Inspects available datasets in data_dir and returns their availability and basic file stats.
    """
    target_dir = Path(data_dir) if data_dir else get_default_data_dir()
    inventory = {}
    for name in ["cm1", "jm1"]:
        p = target_dir / f"{name}.csv"
        if p.exists():
            inventory[name] = {
                "available": True,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "size_kb": round(p.stat().st_size / 1024, 2)
            }
        else:
            inventory[name] = {
                "available": False,
                "path": str(p),
                "size_bytes": 0,
                "size_kb": 0.0
            }
    return inventory
