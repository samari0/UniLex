"""
Loads and prepares the UniLex dataset for search and lookup.
"""
import pandas as pd
from pathlib import Path
import re

DATASET_PATH = Path(__file__).resolve().parent / "unilex_dataset.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # strip BOM if present

    required = [
        "term", "formal_definition", "student_friendly_explanation",
        "example", "category", "difficulty", "related_terms", "source"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in required:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if df["term"].eq("").any() or df["formal_definition"].eq("").any():
        raise ValueError("Terms and definitions must not be empty.")
    df["term_lower"] = df["term"].str.casefold()
    if df["term_lower"].duplicated().any():
        raise ValueError("Duplicate dictionary terms require review.")
    df = df.reset_index(drop=True)
    df["id"] = df.index
    return df


def build_search_text(row: pd.Series) -> str:
    """Text blob used for TF-IDF / SBERT indexing — combines all searchable fields."""
    parts = [
        row["term"],
        row["formal_definition"],
        row["student_friendly_explanation"],
        row["example"],
    ]
    return " ".join(p for p in parts if p)


def to_entry_dict(row: pd.Series) -> dict:
    """Public-facing shape for a single dictionary entry."""
    related = [t.strip() for t in row["related_terms"].split(";") if t.strip()]
    return {
        "term": row["term"],
        "formal_definition": row["formal_definition"],
        "student_friendly_explanation": row["student_friendly_explanation"],
        "example": row["example"],
        "category": row["category"],
        "difficulty": row["difficulty"],
        "related_terms": related,
        "source": row["source"],
    }
