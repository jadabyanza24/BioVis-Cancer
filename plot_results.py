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
    ax.set_title('Stage distribution: ' + ('historical experiment' if historical else 'revised cohort'), loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    ax.set_xlabel('Recorded stage')
    ax.set_ylabel('Number of ' + ('RNA-seq files' if historical else 'unique cases'))
    ax.set_ylim(0, max(counts) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    note = ('Source: saved historical notebook chart; number of unique patients is unverified.' if historical else
            'One primary tumor file per case; selection and labels require further audit.')
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
           xlabel='Predicted stage', ylabel='Recorded stage')
    ax.set_title('Confusion matrix: ' + ('historical experiment' if historical else 'revised evaluation'), loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    fig.colorbar(im, ax=ax, label='Number of cases', fraction=.046, pad=.04)
    note = ('Historical evaluation has data leakage and is not a reliable performance estimate.' if historical else
            'Exploratory evaluation; parameters were informed by earlier work. Not clinical validation.')
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
    ax.set_title('Global feature importance: top 10 genes', loc='left', fontsize=15, weight='bold', color=DARK, pad=16)
    ax.set_xlabel('Model feature importance')
    ax.set_ylabel('Gene ID')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.08, -0.06, 'Global model importance, not a patient-specific explanation or evidence of causation.', color='#594B52', fontsize=9)
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
