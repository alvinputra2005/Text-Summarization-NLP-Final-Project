from unittest import TestCase
from unittest.mock import patch

from src.contextual_cleaning import checked_edit, contextual_parts, refine_context
from src.gemini_preprocessing import GeminiConfig


class ContextualCleaningTests(TestCase):
    def test_referent_needs_both_neighbor_and_verified_code(self):
        row = {"text": "Ubah yang ini menjadi 3.", "before": "Gunakan variabel batas.", "after": "", "code": ["batas = 3"]}
        edit = {"kind": "referent", "quote": "yang ini", "replacement": "batas", "evidence": "batas = 3"}
        self.assertEqual(checked_edit(row, edit), "Ubah `batas` menjadi 3.")
        self.assertIsNone(checked_edit({**row, "before": "Gunakan variabel lain."}, edit))
        self.assertIsNone(checked_edit(row, {**edit, "evidence": "batas = 30"}))

    def test_repeat_and_pause_can_be_removed_without_rewriting(self):
        row = {"text": "Lalu, anu, kita simpan simpan hasil.", "before": "Nilainya sudah tampil.", "after": "", "code": []}
        edit = {"kind": "remove", "replacement": "", "evidence": row["before"]}
        self.assertEqual(checked_edit(row, {**edit, "quote": "anu"}), "Lalu, kita simpan simpan hasil.")
        self.assertEqual(checked_edit(row, {**edit, "quote": "simpan"}), "Lalu, anu, kita simpan hasil.")
        for wording, span in (("Jawab `ya` saat muncul peringatan.", "ya"), ("Kita tidak tidak berhenti.", "tidak"),
                              ("Masukkan 3 3 jika diminta.", "3"), ("Apa bedanya, anu, di sini?", "anu")):
            with self.subTest(wording=wording):
                self.assertIsNone(checked_edit({**row, "text": wording}, {**edit, "quote": span}))

    def test_changes_preserve_the_source_view_and_timestamps(self):
        trace = [{"start": 1, "cleaned": "Gunakan batas.", "original": "Gunakan batas."},
                 {"start": 4, "cleaned": "Ubah yang ini menjadi 3.", "original": "Ubah yang ini menjadi 3."}]
        part = {"id": 1, "trace": trace, "frames": [{"id": "awal", "second": 4}, {"id": "akhir", "second": 5}]}
        observation = {"part": 1, "frame_ids": ["awal", "akhir"], "code": "batas = 3"}
        edit = {"segment": 1, "kind": "referent", "quote": "yang ini", "replacement": "batas", "evidence": "batas = 3"}
        with patch("src.contextual_cleaning.request_gemini_json", return_value=({"edits": [edit]}, {})) as send:
            result = refine_context([part], [observation], GeminiConfig("test"))
        self.assertEqual(send.call_count, 1)
        revised = contextual_parts([part], result)[0]["trace"][1]
        self.assertEqual(revised["cleaned"], "Ubah `batas` menjadi 3.")
        self.assertEqual((revised["start"], revised["original"]), (4, trace[1]["original"]))
        self.assertEqual(trace[1]["cleaned"], trace[1]["original"])
