from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def prepare_expression(frame, training_genes):
    """Validate and align one sample to the scaler's exact feature names."""
    required = {'gene_id', 'fpkm_unstranded'}
    if not required.issubset(frame.columns):
        raise ValueError('The file must contain gene_id and fpkm_unstranded columns.')
    genes = frame.loc[frame['gene_id'].astype(str).str.startswith('ENSG'), ['gene_id', 'fpkm_unstranded']].copy()
    if genes.empty or genes['gene_id'].duplicated().any():
        raise ValueError('ENSG gene IDs are missing or duplicated.')
    genes['fpkm_unstranded'] = pd.to_numeric(genes['fpkm_unstranded'], errors='coerce')
    values = genes['fpkm_unstranded'].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > np.finfo(np.float32).max).any():
        raise ValueError('Expression values must be finite, non-negative, and within the float32 range.')
    expected = pd.Index(training_genes)
    if expected.has_duplicates or expected.empty:
        raise ValueError('The scaler feature list is empty or contains duplicates.')
    series = genes.set_index('gene_id')['fpkm_unstranded']
    missing = expected.difference(series.index)
    if len(missing):
        raise ValueError(f'{len(missing)} required genes are missing. Gene IDs must match exactly, including versions.')
    return series.reindex(expected).to_frame().T
