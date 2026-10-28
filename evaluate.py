"""Leakage-aware research evaluation for the downloaded TCGA-LUAD cohort."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn import __version__ as imblearn_version
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from xgboost import XGBClassifier

DATA = Path('data/gdc')
OUTPUT = Path('outputs')


def load_cohort(data_dir=DATA):
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    if len({x['case_id'] for x in manifest}) != len(manifest):
        raise ValueError('Manifest contains duplicate cases')
    columns = []
    expected = None
    for item in manifest:
        source = data_dir / (item['file_id'] + '.tsv')
        if 'md5' not in item or 'size' not in item:
            raise ValueError('Manifest has no checksums. Run the latest download_gdc.py to verify the cohort.')
        content = source.read_bytes()
        if len(content) != item['size'] or hashlib.md5(content).hexdigest() != item['md5']:
            raise ValueError(f"GDC checksum or file size mismatch: {item['file_id']}")
        df = pd.read_csv(source, sep='\t', comment='#', usecols=['gene_id', 'fpkm_unstranded'])
        df = df[df['gene_id'].str.startswith('ENSG', na=False)].set_index('gene_id')
        if not df.index.is_unique:
            raise ValueError(f"Duplicate gene IDs: {item['file_id']}")
        values = pd.to_numeric(df['fpkm_unstranded'], errors='raise')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid expression values: {item['file_id']}")
        if expected is None:
            expected = values.index
        if not values.index.equals(expected):
            raise ValueError(f"Gene IDs or order differ: {item['file_id']}")
        columns.append(values.to_numpy(dtype=np.float32))
    X = pd.DataFrame(np.stack(columns), columns=expected, index=[x['case_id'] for x in manifest])
    y = np.array([x['stage'] for x in manifest])
    return X, y, manifest


def log_expression(data):
    # Match the cohort's float32 storage, then compute in float64 to avoid
    # spurious ANOVA scores from floating-point cancellation.
    return np.log1p(data.astype(np.float32).astype(np.float64))


def build_pipeline():
    return Pipeline([
        ('log', FunctionTransformer(log_expression, feature_names_out='one-to-one')),
        ('scale', QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)),
        ('variance', VarianceThreshold()),
        ('anova', SelectKBest(f_classif, k=500)),
        ('rfe', RFE(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1), n_features_to_select=150, step=0.1)),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.2, subsample=0.8,
                                objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=1)),
    ])


def main():
    OUTPUT.mkdir(exist_ok=True)
    X, y, manifest = load_cohort()
    if set(y) != {0, 1, 2, 3} or min(np.bincount(y)) < 12:
        raise ValueError('Need all four classes with at least 12 samples each for the configured split and SMOTE folds.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert set(X_train.index).isdisjoint(X_test.index)
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print(f'Cohort: {len(X)} cases, {X.shape[1]} genes. Train: {len(X_train)}, test: {len(X_test)}.', flush=True)
    print('Running three-fold cross-validation on training cases...', flush=True)
    cv_result = cross_validate(pipe, X_train, y_train, cv=cv, scoring={'macro_f1': 'f1_macro', 'accuracy': 'accuracy'}, n_jobs=1, error_score='raise')
    print('Fitting on training cases and evaluating held-out cases...', flush=True)
    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    baseline_prediction = baseline.predict(X_test)
    report = classification_report(y_test, prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    result = {
        'description': 'Exploratory held-out evaluation; not clinical validation. Parameters were informed by a prior leaky experiment, so this is not independent confirmation.',
        'dataset': 'GDC TCGA-LUAD primary tumor RNA-Seq STAR Counts; one file per case',
        'manifest_sha256': hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
        'manifest_acquired_utc': datetime.fromtimestamp((DATA / 'manifest.json').stat().st_mtime, timezone.utc).isoformat(),
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'cases': len(X), 'genes': X.shape[1], 'class_counts': np.bincount(y).tolist(),
        'train_cases': len(X_train), 'test_cases': len(X_test),
        'cv_macro_f1': cv_result['test_macro_f1'].tolist(), 'cv_accuracy': cv_result['test_accuracy'].tolist(),
        'test_report': report,
        'majority_baseline_report': classification_report(y_test, baseline_prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, prediction, labels=[0, 1, 2, 3]).tolist(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
        'imbalanced_learn': imblearn_version,
        'train_case_ids': X_train.index.tolist(), 'test_case_ids': X_test.index.tolist(),
        'file_ids': [item['file_id'] for item in manifest],
    }
    # Publish only a fully written model; pair the report with its exact bytes.
    model_path = OUTPUT / 'research_pipeline.joblib'
    temporary_model = OUTPUT / 'research_pipeline.joblib.part'
    joblib.dump(pipe, temporary_model, compress=3)
    result['model_sha256'] = hashlib.sha256(temporary_model.read_bytes()).hexdigest()
    temporary_model.replace(model_path)
    temporary_report = OUTPUT / 'evaluation.json.part'
    temporary_report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary_report.replace(OUTPUT / 'evaluation.json')
    print(classification_report(y_test, prediction, labels=[0, 1, 2, 3], zero_division=0), flush=True)
    print('Saved research pipeline and evaluation in outputs/.', flush=True)

if __name__ == '__main__':
    main()
