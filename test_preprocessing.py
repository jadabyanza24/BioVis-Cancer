import unittest
import numpy as np
import pandas as pd
from preprocessing import prepare_expression

class InputValidationTests(unittest.TestCase):
    def test_aligns_exact_feature_order(self):
        frame = pd.DataFrame({'gene_id': ['ENSG2.1', 'ENSG1.1'], 'fpkm_unstranded': [3, 1]})
        result = prepare_expression(frame, ['ENSG1.1', 'ENSG2.1'])
        np.testing.assert_allclose(result.to_numpy(), [[1, 3]])

    def test_rejects_missing_duplicate_and_invalid(self):
        cases = [
            pd.DataFrame({'gene_id': ['ENSG1.1'], 'fpkm_unstranded': [1]}),
            pd.DataFrame({'gene_id': ['ENSG1.1', 'ENSG1.1', 'ENSG2.1'], 'fpkm_unstranded': [1, 2, 3]}),
            pd.DataFrame({'gene_id': ['ENSG1.1', 'ENSG2.1'], 'fpkm_unstranded': [1, -1]}),
            pd.DataFrame({'gene_id': ['ENSG1.1', 'ENSG2.1'], 'fpkm_unstranded': [1, float('nan')]}),
        ]
        for frame in cases:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                prepare_expression(frame, ['ENSG1.1', 'ENSG2.1'])

if __name__ == '__main__':
    unittest.main()
