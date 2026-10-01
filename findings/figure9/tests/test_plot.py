"""Check complete sample coverage and paired values in both Figure 9 layouts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


FIGURE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('loopvl_figure9_plot', FIGURE_ROOT / 'plot.py')
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class Figure9PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((FIGURE_ROOT / 'data/lh_endpoint_data.json').read_text(encoding='utf-8'))

    def check_layout(self, entropy_only):
        expected = [self.data['L'][name]['pairs'] for name in ('L1', 'L2', 'L3')]
        expected.append(self.data['H']['response_entropy']['pairs'])
        if not entropy_only:
            expected.extend(self.data['H'][name]['pairs'] for name in ('response_coverage80', 'sink_gini'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'figure.pdf'
            preview = Path(directory) / 'figure.png'
            # Keep the generated axes accessible for row-by-row assertions.
            with patch.object(PLOT.plt, 'close') as close:
                audit = PLOT.draw(self.data, entropy_only, output, preview)
                figure = close.call_args.args[0]
            try:
                self.assertEqual(audit['sample_order'], self.data['sample_order'])
                self.assertEqual(len(figure.axes), len(expected))
                self.assertEqual(len(audit['panels']), len(expected))
                self.assertTrue(output.is_file())
                self.assertTrue(preview.is_file())
                for axis, panel, values in zip(figure.axes, audit['panels'], expected):
                    values = np.asarray(values, dtype=float)
                    self.assertEqual(len(axis.lines), 33)
                    for line, row in zip(axis.lines[:32], values):
                        np.testing.assert_array_equal(line.get_xdata(), [1, 2])
                        np.testing.assert_array_equal(line.get_ydata(), row)
                    np.testing.assert_array_equal(axis.lines[32].get_ydata(), values.mean(axis=0))
                    np.testing.assert_array_equal(panel['mean'], values.mean(axis=0))
                    for key in ('paired_samples', 'sample_lines', 'displayed_sample_count', 'mean_sample_count'):
                        self.assertEqual(panel[key], 32)
                self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()], ['Samples', 'Mean'])
            finally:
                PLOT.plt.close(figure)

    def test_six_panel_layout_preserves_every_sample(self):
        self.check_layout(False)

    def test_entropy_layout_preserves_every_sample(self):
        self.check_layout(True)

    def test_sample_ids_are_unique(self):
        data = copy.deepcopy(self.data)
        data['sample_order'][1] = data['sample_order'][0]
        with self.assertRaises(AssertionError):
            PLOT.draw(data, False, None, None)

    def test_sample_count_matches_sample_order(self):
        data = copy.deepcopy(self.data)
        data['sample_order'].pop()
        with self.assertRaises(AssertionError):
            PLOT.draw(data, False, None, None)


if __name__ == '__main__':
    unittest.main()
