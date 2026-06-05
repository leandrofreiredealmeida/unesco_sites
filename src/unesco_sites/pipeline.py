from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from unesco_sites import logger
from unesco_sites.data.loader import load_processed_data
from unesco_sites.models.similarity import build_tfidf, classify_thematic


@dataclass(frozen=True)
class PipelineState:
    """Immutable snapshot of all data and models ready for the dashboard."""

    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: csr_matrix


def build() -> PipelineState:
    """Runs the full data and modelling pipeline in sequence.

    Steps:
        1. Load processed CSV via loader.
        2. Filter rows with valid full_text and coordinates.
        3. Fit TF-IDF vectorizer.
        4. Assign primary thematic category to every site.

    Returns:
        PipelineState with the enriched DataFrame, fitted vectorizer,
        and TF-IDF sparse matrix (all row-aligned).
    """
    logger.info("=== Pipeline started ===")

    df = load_processed_data()

    df = df.dropna(subset=["full_text", "latitude", "longitude"]).reset_index(drop=True)
    logger.info("Sites with full_text and coordinates: %d", len(df))

    vectorizer, tfidf_matrix = build_tfidf(df)

    df = classify_thematic(df, tfidf_matrix, vectorizer)

    logger.info("=== Pipeline complete: %d sites ready ===", len(df))
    return PipelineState(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)
