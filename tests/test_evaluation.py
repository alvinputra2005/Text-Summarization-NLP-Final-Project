import unittest

from src.code_text import highlight_code, technical_terms
from src.evaluation import rouge_breakdown, word_errors


class EvaluationTests(unittest.TestCase):
    def test_highlights_functions_and_declared_variables_and_escapes_html(self):
        text = 'variabel jawaban = input(). Ubah jawaban dengan int() <b>'
        rendered = highlight_code(text)
        self.assertEqual(rendered.count('<code>jawaban</code>'), 2)
        self.assertIn('<code>input()</code>', rendered)
        self.assertIn('<code>int()</code>', rendered)
        self.assertTrue(rendered.endswith('&lt;b&gt;'))
        self.assertEqual(technical_terms(text), {'jawaban', 'input', 'int'})

    def test_rouge_components_and_trigrams_use_the_same_reference(self):
        scores = rouge_breakdown('baca data csv', 'baca data csv baru')
        self.assertAlmostEqual(scores['rouge3']['Precision'], .5)
        self.assertAlmostEqual(scores['rouge3']['Recall'], 1)
        self.assertAlmostEqual(scores['rouge3']['F1'], 2 / 3)

    def test_word_errors_groups_changes_by_edit_operation(self):
        self.assertEqual(word_errors('Baca data CSV.', 'Baca berkas CSV.'), {
            'Penggantian': 1, 'Penghapusan': 0, 'Penyisipan': 0,
        })
