from __future__ import annotations

from pathlib import Path

import pandas as pd

from unesco_sites import logger

_PROCESSED = (
    Path(__file__).parents[3] / "data" / "processed" / "whc_processed.csv"
)


def load_processed_data() -> pd.DataFrame:
    """Loads the processed UNESCO WHC dataset from disk.

    Returns:
        DataFrame with 1 248 sites and 46 columns.

    Raises:
        FileNotFoundError: If the processed CSV is not found at the expected path.
    """
    if not _PROCESSED.exists():
        raise FileNotFoundError(f"Processed dataset not found: {_PROCESSED}")

    logger.info("Loading processed data from %s", _PROCESSED)
    df = pd.read_csv(_PROCESSED, sep=";", encoding="utf-8")
    logger.info("Data loaded: %d rows x %d columns", df.shape[0], df.shape[1])
    return df
