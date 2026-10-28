import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import plot_results

class HistoricalFigureProvenanceTests(unittest.TestCase):
    def test_transcribed_confusion_matches_saved_xgboost_report(self):
        notebook = json.loads(Path('notebooks/historical_experiment.ipynb').read_text(encoding='utf-8'))
        streams = [''.join(o.get('text', [])) for c in notebook['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'stream']
        report = next(s for s in streams if 'Classification Report Final (RFE + Tuned XGBoost)' in s)
        with patch.object(plot_results, 'distribution') as distribution, patch.object(plot_results, 'confusion') as confusion:
            plot_results.historical_figures(Path('.tmp'))
        counts = distribution.call_args.args[0]
        matrix = np.array(confusion.call_args.args[0])
        self.assertEqual(sum(counts), 394)
        self.assertEqual(matrix.sum(), 79)
        for i, stage in enumerate(['I', 'II', 'III', 'IV']):
            row = re.search(rf'^\s*Stadium {stage}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$', report, re.M)
            self.assertIsNotNone(row)
            precision, recall, f1, support = row.groups()
            tp = matrix[i, i]
            actual_precision = tp / matrix[:, i].sum()
            actual_recall = tp / matrix[i, :].sum()
            actual_f1 = 2 * actual_precision * actual_recall / (actual_precision + actual_recall)
            self.assertAlmostEqual(actual_precision, float(precision), delta=.0051)
            self.assertAlmostEqual(actual_recall, float(recall), delta=.0051)
            self.assertAlmostEqual(actual_f1, float(f1), delta=.0051)
            self.assertEqual(matrix[i, :].sum(), int(support))

if __name__ == '__main__':
    unittest.main()
