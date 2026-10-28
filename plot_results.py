"""Create transparent-to-review figures from recorded experiment results.

Historical values were transcribed from saved figures in historical_experiment.ipynb;
they have not been recomputed from the original dataset. Revised figures require
an evaluation.json produced by evaluate.py. Neither set is clinical evidence.
"""
import hashlib
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = ['I', 'II', 'III', 'IV']
PINK = '#C83F69'
DARK = '#32252C'


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def distribution(counts, path, *, historical):
    counts = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
    fig.patch.set_facecolor('white')
    bars = ax.bar(LABELS, counts, color=PINK, width=0.62)
    ax.bar_label(bars, padding=5, fontsize=11, color=DARK)
    ax.set_title('Distribusi stadium — ' + ('eksperimen historis' if historical else 'kohort evaluasi ulang'), loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    ax.set_xlabel('Stadium tercatat')
    ax.set_ylabel('Jumlah ' + ('file RNA-seq' if historical else 'kasus unik'))
    ax.set_ylim(0, max(counts) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    note = ('Sumber: grafik tersimpan di notebook historis; jumlah pasien unik belum diverifikasi.' if historical else
            'Satu file tumor primer dipilih per kasus; pemilihan dan label perlu diaudit lebih lanjut.')
    fig.text(0.08, -0.08, note, color='#594B52', fontsize=9, wrap=True)
    _save(fig, path)


def confusion(matrix, path, *, historical):
    matrix = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 5.5), layout='constrained')
    im = ax.imshow(matrix, cmap='RdPu', vmin=0)
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha='center', va='center', fontsize=12,
                color='white' if count > matrix.max() * .55 else DARK, weight='bold')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Prediksi model', ylabel='Stadium tercatat')
    ax.set_title('Confusion matrix — ' + ('eksperimen historis' if historical else 'evaluasi ulang'), loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    fig.colorbar(im, ax=ax, label='Jumlah sampel', fraction=.046, pad=.04)
    note = ('Evaluasi historis mengalami data leakage; bukan estimasi kinerja yang andal.' if historical else
            'Evaluasi eksploratif; parameter dipengaruhi eksperimen lama. Bukan validasi klinis.')
    fig.text(0.08, -0.06, note, color='#594B52', fontsize=9, wrap=True)
    _save(fig, path)


def historical_figures(destination=Path('figures')):
    # Values transcribed from the original saved class chart and confusion matrix.
    distribution([215, 90, 65, 24], destination / 'historical_class_distribution.png', historical=True)
    confusion([[38, 3, 1, 1], [9, 4, 5, 0], [6, 0, 7, 0], [1, 3, 0, 1]],
              destination / 'historical_confusion_matrix.png', historical=True)


def global_importance(model_path, path):
    # Only load a locally generated, trusted artifact: joblib deserialization executes code.
    pipeline = joblib.load(model_path)
    genes = pipeline.named_steps['scale'].feature_names_in_
    genes = genes[pipeline.named_steps['variance'].get_support()][pipeline.named_steps['anova'].get_support()][pipeline.named_steps['rfe'].get_support()]
    weights = pipeline.named_steps['model'].feature_importances_
    top = np.argsort(weights)[-10:]
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    ax.barh(genes[top], weights[top], color=PINK)
    ax.set_title('Feature importance global — 10 gen teratas', loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    ax.set_xlabel('Feature importance model')
    ax.set_ylabel('ID gen')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.08, -0.06, 'Importance global model, bukan penjelasan pasien atau bukti sebab-akibat.', color='#594B52', fontsize=9)
    _save(fig, path)


def revised_figures(result_path=Path('outputs/evaluation.json'), destination=Path('figures'), model_path=Path('outputs/research_pipeline.joblib')):
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != result.get('model_sha256'):
        raise ValueError('Model does not match the evaluation report.')
    distribution(result['class_counts'], destination / 'revised_class_distribution.png', historical=False)
    confusion(result['confusion_matrix'], destination / 'revised_confusion_matrix.png', historical=False)
    global_importance(model_path, destination / 'revised_global_importance.png')

if __name__ == '__main__':
    historical_figures()
