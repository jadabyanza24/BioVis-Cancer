import hashlib
import json
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from evaluate import load_cohort, build_pipeline, log_expression

class CohortValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.directory.name)
        self.data = b'gene_id\tfpkm_unstranded\nENSG0001.1\t2.5\nENSG0002.1\t0\n'
        (self.root / 'one.tsv').write_bytes(self.data)
        self.row = {'case_id': 'one', 'file_id': 'one', 'stage': 0,
                    'size': len(self.data), 'md5': hashlib.md5(self.data).hexdigest()}

    def tearDown(self):
        self.directory.cleanup()

    def manifest(self, rows):
        (self.root / 'manifest.json').write_text(json.dumps(rows), encoding='utf-8')

    def test_reads_verified_single_case(self):
        self.manifest([self.row])
        x, y, _ = load_cohort(self.root)
        self.assertEqual(x.index.tolist(), ['one'])
        self.assertEqual(y.tolist(), [0])
        self.assertEqual(x.iloc[0].tolist(), [2.5, 0.0])

    def test_rejects_corrupt_file_and_duplicate_cases(self):
        self.manifest([{**self.row, 'md5': '0' * 32}])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_cohort(self.root)
        self.manifest([self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'duplicate cases'):
            load_cohort(self.root)

class ConstantFeatureRegressionTests(unittest.TestCase):
    def test_inference_matches_float32_cohort_storage(self):
        raw = pd.DataFrame({'ENSG1.1': [7.801123456789]})
        np.testing.assert_array_equal(log_expression(raw), log_expression(raw.astype(np.float32)))

    def test_constant_genes_are_removed_after_float64_scaling(self):
        rng = np.random.default_rng(5)
        data = np.full((20, 510), 7.8011, dtype=np.float32)
        data[:, :200] = rng.uniform(0, 20, (20, 200))
        frame = pd.DataFrame(data, columns=[f'ENSG{i:011}.1' for i in range(510)])
        pipeline = build_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logged = pipeline.named_steps['log'].fit_transform(frame)
            self.assertEqual(logged.to_numpy().dtype, np.float64)
            scaled = pipeline.named_steps['scale'].fit_transform(logged)
            filtered = pipeline.named_steps['variance'].fit_transform(scaled)
        self.assertEqual(filtered.shape[1], 200)
        self.assertTrue((np.std(filtered, axis=0) > 0).all())

if __name__ == '__main__':
    unittest.main()
