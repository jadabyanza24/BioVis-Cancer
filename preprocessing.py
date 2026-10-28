from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def prepare_expression(frame, training_genes):
    """Validate and align one sample to the scaler's exact feature names."""
    required = {'gene_id', 'fpkm_unstranded'}
    if not required.issubset(frame.columns):
        raise ValueError('File harus memiliki kolom gene_id dan fpkm_unstranded.')
    genes = frame.loc[frame['gene_id'].astype(str).str.startswith('ENSG'), ['gene_id', 'fpkm_unstranded']].copy()
    if genes.empty or genes['gene_id'].duplicated().any():
        raise ValueError('ID gen ENSG kosong atau duplikat.')
    genes['fpkm_unstranded'] = pd.to_numeric(genes['fpkm_unstranded'], errors='coerce')
    values = genes['fpkm_unstranded'].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > np.finfo(np.float32).max).any():
        raise ValueError('Nilai ekspresi harus berupa angka terhingga, tidak negatif, dan dalam rentang float32.')
    expected = pd.Index(training_genes)
    if expected.has_duplicates or expected.empty:
        raise ValueError('Daftar fitur scaler kosong atau mengandung duplikat.')
    series = genes.set_index('gene_id')['fpkm_unstranded']
    missing = expected.difference(series.index)
    if len(missing):
        raise ValueError(f'{len(missing)} gen yang diperlukan tidak ditemukan. ID gen harus cocok persis, termasuk versi.')
    return series.reindex(expected).to_frame().T
