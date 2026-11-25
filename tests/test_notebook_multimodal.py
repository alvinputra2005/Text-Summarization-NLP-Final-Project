from dataclasses import asdict
from unittest import TestCase
from unittest.mock import patch

from scripts.notebook_multimodal import (
    OCRLine, clean_cues, confirm_gemini, consensus, contains_condition, find_evidence, gemini_vision,
)
from src.gemini_preprocessing import GeminiConfig
from src.segmentation import Segment
from src.transcript_cleaning import clean_utterance


def frame(second, action, condition="if angka == 3:"):
    lines = [
        OCRLine("while angka < 5:", 16, 10, 95),
        OCRLine(condition, 65, 30, 94),
        OCRLine(action, 99, 42, 91),
    ]
    return {"frame_id": f"frame-{second}", "second": second, "observations": [asdict(item) for item in find_evidence(lines)],
            "jpeg": "Zml4dHVyZQ=="}


class MultimodalEvidenceTest(TestCase):
    def test_removes_fillers_only_where_they_are_clearly_discourse_markers(self):
        cases = (
            ("Nah kita buka file CSV.", "kita buka file CSV.", ["Nah"]),
            ("Oke, sekarang jalankan print().", "sekarang jalankan print().", ["Oke"]),
            ("Lalu, apa ya, kita jalankan lagi.", "Lalu, kita jalankan lagi.", ["apa ya"]),
            ("Ini kondisi pertama, oke ini contoh berikutnya.", "Ini kondisi pertama, ini contoh berikutnya.", ["oke"]),
            ("Kondisinya siap, oke, nah kalau kosong, simpan.", "Kondisinya siap, kalau kosong, simpan.", ["oke", "nah"]),
            ("Apa yang terjadi kalau input kosong?", "Apa yang terjadi kalau input kosong?", []),
            ("Apa ya?", "Apa ya?", []),
            ("Ya.", "Ya.", []),
            ("Oke?", "Oke?", []),
            ("Statusnya oke.", "Statusnya oke.", []),
            ("Beri status, oke, kemudian simpan.", "Beri status, oke, kemudian simpan.", []),
            ("Pakai 4, eh, maksudnya 5.", "Pakai 4, eh, maksudnya 5.", []),
            ('Baca print("jadi, apa ya, kita ulang") dan `oke = False`.',
             'Baca print("jadi, apa ya, kita ulang") dan `oke = False`.', []),
            ("Nah, hmm.", "", ["Nah", "hmm"]),
        )
        for spoken, expected, fillers in cases:
            with self.subTest(spoken=spoken):
                self.assertEqual(clean_utterance(spoken), (expected, fillers))
                self.assertEqual(clean_utterance(expected), (expected, []))

    def test_cleaning_audit_retains_timestamps_and_marks_overlapping_duplicates(self):
        segments = [Segment(1, 4, "Nah kita cek input."), Segment(3, 5, "kita cek input."),
                    Segment(5, 6, "Eh."), Segment(6, 8, "Oke?")]
        rows = clean_cues(segments, 0, 9)
        self.assertEqual([row["status"] for row in rows],
                         ["dirapikan", "duplikat tumpang tindih", "pengisi mandiri", "dipertahankan"])
        self.assertEqual([(row["start"], row["end"], row["original"]) for row in rows],
                         [(cue.start, cue.end, cue.transcript) for cue in segments])
        self.assertEqual(rows[0]["fillers_removed"], ["Nah"])

    def test_cleaning_keeps_negation_identifiers_and_original(self):
        source = " Nah, nggak jalanin saat gak = 3 dan parameter nggak dipakai. "
        rows = clean_cues([Segment(1, 4, source)], 0, 7)
        self.assertEqual(rows[0]["original"], source)
        self.assertEqual(rows[0]["cleaned"], "tidak menjalankan saat gak = 3 dan parameter nggak dipakai.")

    def test_commented_break_never_counts_and_requires_two_active_frames(self):
        frames = [frame(310, "# break"), frame(315, "break"), frame(318, "break")]
        self.assertFalse(frames[0]["observations"][0]["active"])
        self.assertIsNone(consensus(frames[:2]))
        self.assertEqual(consensus(frames)["frames"], ["frame-315", "frame-318"])
        self.assertEqual(consensus(frames)["expression"], "angka == 3")
        self.assertIsNone(consensus([frame(315, "break"), frame(318, "break", "if angka = 3:")]))
        self.assertIsNone(consensus([frame(315, "break"), frame(318, "break", "if angka == 4:")]))

    def test_condition_check_rejects_partial_numbers_and_actions_in_nested_blocks(self):
        self.assertTrue(contains_condition("Saat angka==3.", "angka == 3"))
        self.assertFalse(contains_condition("Saat angka == 30.", "angka == 3"))
        self.assertFalse(contains_condition("Saat angka == 3.5.", "angka == 3"))
        self.assertFalse(find_evidence([
            OCRLine("while angka < 5:", 16, 10, 95),
            OCRLine("if angka == 3:", 65, 30, 94),
            OCRLine("if siap:", 99, 42, 91),
            OCRLine("break", 143, 52, 96),
        ]))

    def test_multimodal_request_uses_crops_or_bounded_video_and_validates_response(self):
        frames = [frame(315, "break"), frame(318, "break")]
        trace = [{"cleaned": "Kalau ini nilainya tiga, kondisi break terpenuhi."}]
        response = {"speech_quote": trace[0]["cleaned"], "condition": "if angka == 3:", "referent": "angka",
                    "action": "break", "frame_ids": ["frame-315", "frame-318"]}
        with patch("scripts.notebook_multimodal.gemini_config", return_value=GeminiConfig("test-key")), patch(
            "scripts.notebook_multimodal.request_gemini_json", return_value=(response, {"totalTokenCount": 123}),
        ) as request:
            image_result = gemini_vision(frames, trace, 312, 320, "frames", "")
            content = request.call_args.args[1]["contents"][0]["parts"]
            self.assertEqual(sum("inlineData" in part for part in content), 2)
            video_result = gemini_vision(frames, trace, 312, 320, "youtube", "https://youtu.be/B6scLunzn0I")
            media = request.call_args.args[1]["contents"][0]["parts"][-1]
            self.assertEqual(media["videoMetadata"]["endOffset"], "320.0s")
        self.assertEqual(image_result["usage"], {"totalTokenCount": 123})
        self.assertTrue(confirm_gemini(image_result, consensus(frames), trace))
        self.assertTrue(confirm_gemini(video_result, consensus(frames), trace))
        response["condition"] = "if angka == 9:"
        self.assertFalse(confirm_gemini(image_result, consensus(frames), trace))
