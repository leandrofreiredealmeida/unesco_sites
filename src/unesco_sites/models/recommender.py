from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel

from unesco_sites import logger


def get_recommendations(
    site_idx: int,
    df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    *,
    top_n: int = 10,
    geo_weight: float = 0.5,
) -> pd.DataFrame:
    """Returns the top_n UNESCO sites most similar to the given site.

    Combines two normalised scores into a single ranking:
    - Geographic score: inverse of haversine distance (closer = higher).
    - NLP score: TF-IDF cosine similarity via linear_kernel.

    Args:
        site_idx: Integer index of the query site in df.
        df: DataFrame aligned row-by-row with tfidf_matrix.
        tfidf_matrix: Sparse matrix (n_sites x n_terms).
        top_n: Number of recommendations to return.
        geo_weight: Weight assigned to geographic proximity in [0, 1].
            0 = pure NLP similarity, 1 = pure geographic proximity.

    Returns:
        DataFrame with top_n rows and columns:
        name_en, category_thematic, category, region, year_inscribed,
        geo_dist_km, nlp_sim, score.
    """
    logger.info(
        "Computing recommendations for idx=%d (top_n=%d, geo_weight=%.2f)",
        site_idx,
        top_n,
        geo_weight,
    )

    # ── Geographic score (haversine) ─────────────────────────────────────────
    site_lat_rad = np.radians(df.loc[site_idx, "latitude"])
    site_lon_rad = np.radians(df.loc[site_idx, "longitude"])
    coords_rad = np.radians(df[["latitude", "longitude"]].values)

    dlat = coords_rad[:, 0] - site_lat_rad
    dlon = coords_rad[:, 1] - site_lon_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(site_lat_rad) * np.cos(coords_rad[:, 0]) * np.sin(dlon / 2) ** 2
    )
    geo_dist_rad = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    geo_dist_km = geo_dist_rad * 6_371.0

    max_dist = geo_dist_rad.max()
    geo_score = 1.0 - geo_dist_rad / (max_dist + 1e-9)

    # ── NLP similarity score ─────────────────────────────────────────────────
    nlp_scores: np.ndarray = linear_kernel(tfidf_matrix[site_idx], tfidf_matrix).flatten()
    max_nlp = nlp_scores.max()
    nlp_norm = nlp_scores / (max_nlp + 1e-9)

    # ── Combined score ────────────────────────────────────────────────────────
    combined = geo_weight * geo_score + (1.0 - geo_weight) * nlp_norm
    combined[site_idx] = -1.0  # exclude the query site itself

    top_idx = combined.argsort()[::-1][:top_n]

    result = df.loc[
        top_idx,
        ["name_en", "category_thematic", "category", "region", "year_inscribed"],
    ].copy()
    result["geo_dist_km"] = geo_dist_km[top_idx].round(0).astype(int)
    result["nlp_sim"] = nlp_scores[top_idx].round(4)
    result["score"] = combined[top_idx].round(4)

    return result.reset_index(drop=True)
