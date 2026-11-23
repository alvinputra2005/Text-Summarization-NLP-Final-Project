from time import perf_counter
from unittest import TestCase
from unittest.mock import Mock, patch

from src.pipeline import finish_guide
from src.segmentation import Segment
from src.step_sources import StepSource
from src.string_sources import is_string_lesson, literal_kinds, string_screen_points
from src.video_context import contextual_sources
from src.video_outline import MAX_GUIDE_POINTS, limit_tutorial_sources


SPEECH = [
    "Kita akan lanjutkan lagi dengan pengenalan string. Kita akan belajar tentang string.",
    "Ini, kan seperti kita tahu sebelumnya, bahwa string ini adalah kubulan dari karakter ya.",
    "Bikin satu, dua, tiga atau dua, tiga ya. Yang pertama itu adalah kita bisa membuat string itu dengan menggunakan single-quad ya.",
    "Kalau dilihat, liat, liat, liat, liat, liat.",
    "Bisa ga kita pakai dua-duanya. Ini kita bisa gabungkan.",
    "Si backslash itu nggak ada. Tampahin backslash lagi double.",
    "Ini pakai smaring slash n.",
    "Yang bawa, ini dipakain oleh Windows. Di pakai oleh Windows.",
    "Kita akan belajar tentang operasi yang bisa kita lakukan untuk ngitung panjang string.",
]
CUES = [Segment(i * 5.0, i * 5.0 + 4, line) for i, line in enumerate(SPEECH)]


class StringGuideTest(TestCase):
    def test_an_incidental_quote_in_a_formatting_lesson_does_not_change_its_topic(self):
        cues = [Segment(0, 5, "Ini tutorial format string. Buat string di dalam tanda kutip."),
                Segment(6, 12, "Format string ini menampilkan angka menjadi persen.")]
        self.assertFalse(is_string_lesson(cues))

    def test_noisy_string_transcript_returns_supported_points_without_loading_mbart_for_fixed_rules(self):
        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.pipeline.load_cached_mbart",
        ) as load:
            guide = finish_guide(b"mp4", CUES, perf_counter(), 0, Mock(), overview=False)
        load.assert_not_called()
        self.assertEqual(guide.status, "ready")
        self.assertEqual(len(guide.summary_points), 3)
        result = " ".join(guide.summary_points)
        for detail in ("rangkaian karakter", "kutip tunggal", "satu backslash"):
            self.assertIn(detail, result)
        for unsupported in ("liat", "Windows", "smaring", "kutip ganda", "\\n", "\\t"):
            self.assertNotIn(unsupported, result)
        self.assertIn("tanda kutip", guide.overview.text)

    def test_literal_tokens_distinguish_raw_text_escaped_slashes_and_active_escapes(self):
        self.assertIn("newline", literal_kinds(r'x = "satu\ndua"'))
        self.assertIn("tab", literal_kinds(r'x = "satu\tdua"'))
        self.assertIn("backslash", literal_kinds(r'x = "satu\\ndua"'))
        self.assertNotIn("newline", literal_kinds(r'x = "satu\\ndua"'))
        self.assertEqual(literal_kinds(r'x = r"satu\ndua"'), {"raw"})
        self.assertFalse(literal_kinds(r'# x = "satu\ndua"'))
        self.assertIn("escape_quote", literal_kinds(r'x = "Kata \"pagi\""'))

    def test_visible_literal_needs_two_matching_frames_and_preserves_both_escapes(self):
        visible = [r'print("atas\nbawah")', r'print("kiri\tkanan")']
        frames = [{"second": t, "lines": visible} for t in (16, 20)]
        part = {"frames": frames, "speech": "String dengan tanda kutip."}
        self.assertEqual(string_screen_points(CUES, [{**part, "frames": frames[:1]}]), {})
        self.assertEqual(string_screen_points(CUES, [{"frames": [frames[0], {"second": 20, "lines": []}]}]), {})
        sources, _ = contextual_sources(CUES, {"parts": [part], "actions": []}, [])
        points = [step.text for step in limit_tutorial_sources(sources)]
        self.assertTrue(any("\\n" in text for text in points))
        self.assertTrue(any("\\t" in text for text in points))

    def test_guide_bounds_candidates_and_keeps_distinct_verified_steps_at_the_same_time(self):
        generic = [StepSource(cue, f"Langkah {i} dengan istilah {i}.") for i, cue in enumerate(CUES * 3)]
        verified = [StepSource(CUES[0], f"Sisipkan karakter `{escape}` di dalam string.", escape)
                    for escape in ("\\n", "\\t")]
        selected = limit_tutorial_sources([*generic, *verified])
        self.assertLessEqual(len(selected), MAX_GUIDE_POINTS)
        self.assertTrue(set(verified) <= set(selected))
