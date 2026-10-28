import unittest
from download_gdc import stage_label

class StageLabelTests(unittest.TestCase):
    def test_supported_pathological_stages(self):
        for text, expected in [('Stage IA', 0), ('stage IIB', 1), ('Stage IIIA', 2), ('Stage IV', 3), ('Stage IVB', 3)]:
            with self.subTest(text=text):
                self.assertEqual(stage_label(text), expected)

    def test_rejects_unknown_or_other_stages(self):
        for text in [None, '', 'Not Reported', 'Stage 0', 'Stage IIA1', 'Stage X']:
            with self.subTest(text=text):
                self.assertIsNone(stage_label(text))

if __name__ == '__main__':
    unittest.main()
