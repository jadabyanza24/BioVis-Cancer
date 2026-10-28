import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


class AppFlowTests(unittest.TestCase):
    def test_packaged_model_and_sample_are_usable_with_research_warning(self):
        # Force the bundled model to exercise the same path as a fresh checkout.
        source = (ROOT / 'app.py').read_text(encoding='utf-8')
        output_path = "ROOT / 'outputs' / 'research_pipeline.joblib'"
        self.assertIn(output_path, source)
        code = source.replace(
            output_path,
            "ROOT / 'nonexistent-local-output' / 'research_pipeline.joblib'",
        )
        app = AppTest.from_string(code, default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.warning)
        self.assertTrue(any('recall stadium IV' in info.value for info in app.info))
        self.assertTrue(app.button)
        app.button[0].click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.metric)


if __name__ == '__main__':
    unittest.main()
