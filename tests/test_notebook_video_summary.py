import unittest

from scripts.notebook_video_summary import score_video, select_across_video
from src.segmentation import Segment


class NotebookVideoSummaryTest(unittest.TestCase):
    def test_selector_skips_greetings_and_keeps_source_time_for_late_explanation(self):
        cues = [
            Segment(0, 3, "Selamat datang di tutorial Python yang membahas fungsi."),
            Segment(10, 14, "Gunakan fungsi dengan satu argumen untuk membaca data."),
            Segment(88, 95, "Metode ini menghapus kunci dan nilai dari dictionary."),
        ]
        chosen = select_across_video(cues)

        self.assertFalse(any("Selamat datang" in source.text for source in chosen))
        self.assertTrue(any(source.segment == cues[-1] and "dictionary" in source.text for source in chosen))

    def test_time_coverage_requires_matching_terms_inside_the_reference_window(self):
        rows = [{"summary": "Gunakan fungsi untuk membaca data.", "start": 10, "end": 14}]
        fact = {"summary": "Panggil fungsi pembacaan.", "start": 60, "end": 70, "terms": [["fungsi"], ["membaca"]]}
        result = score_video(rows, [fact])

        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["aligned_coverage"], 0)
