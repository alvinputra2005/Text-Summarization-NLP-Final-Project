from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

from scripts.notebook_video_ablation import (
    draft, enrich, evaluate, read_with_gemini, select_transcript, speech_drafts, validated_observations, video_parts,
)
from src.gemini_preprocessing import GeminiConfig
from src.segmentation import Segment


class WholeVideoAblationTest(TestCase):
    def test_long_whisper_segments_are_retried_and_recorded(self):
        video = Path("tutorial.mp4")
        good = [Segment(2, 6, "Gunakan break.")]
        long = [Segment(2, 102, "Gunakan break.")]
        with patch("scripts.notebook_video_ablation.load_transcript", return_value=(good, "fallback")) as load:
            self.assertEqual(select_transcript(good, "adaptasi", video, ""), (good, "adaptasi", None))
            load.assert_not_called()
            cues, model, retry = select_transcript(long, "adaptasi", video, "")
            load.assert_called_once_with(video, None, "")
        self.assertEqual((cues, model, retry["model"], retry["segments"]), (good, "fallback", "adaptasi", 1))

    def test_parts_include_tail_and_assign_each_utterance_once(self):
        cues = [Segment(2, 5, "Pembuka."), Segment(50, 55, "Perulangan."),
                Segment(90, 127, "Pengujian akhir.")]
        parts = video_parts(cues, 130, 55)
        self.assertEqual([(part["start"], part["end"]) for part in parts], [(0, 55), (55, 110), (110, 130)])
        self.assertEqual([row["original"] for part in parts for row in part["trace"]],
                         [cue.transcript for cue in cues])
        self.assertEqual(parts[-1]["trace"], [])

    def test_gemini_needs_an_exact_quote_and_the_same_code_in_two_frames(self):
        code = "if angka == batas:"
        part = {"id": 3, "speech": "Jalankan break ketika mencapai batas.",
                "frames": [{"id": "3-1", "lines": [code]}, {"id": "3-2", "lines": [code]}]}
        observation = {"part": 3, "quote": part["speech"], "code": code, "frame_ids": ["3-1", "3-2"]}
        self.assertEqual(validated_observations({"observations": [observation]}, [part]), [observation])
        for field, value in (("quote", "Kutipan di luar transkrip."), ("code", "if angka != batas:"),
                             ("frame_ids", ["3-1", "3-1"]), ("part", 4)):
            with self.subTest(field=field):
                self.assertEqual(validated_observations({"observations": [{**observation, field: value}]}, [part]), [])

    def test_gemini_sends_two_images_and_returns_verified_context_to_summarizer(self):
        code = "if angka == 3:"
        quote = "Saat angka tiga, gunakan break."
        frames = [{"id": f"1-{i}", "second": i, "lines": [code], "actions": [], "crop": "YWJj"}
                  for i in (1, 2)]
        observation = {"part": 1, "quote": quote, "code": code, "frame_ids": ["1-1", "1-2"]}
        with patch("src.video_context.gemini_config", return_value=GeminiConfig("fake")), patch(
            "src.video_context.request_gemini_json",
            return_value=({"observations": [observation]}, {"totalTokenCount": 52}),
        ) as send:
            status = read_with_gemini([{"id": 1, "speech": quote, "frames": frames}], 1)
            parts = send.call_args.args[1]["contents"][0]["parts"]
        self.assertEqual(sum("inlineData" in part for part in parts), 2)
        self.assertEqual(status["verified"], [observation])
        self.assertEqual(status["token_count"], 52)

    def test_enrichment_does_not_substitute_an_unrelated_numeric_condition(self):
        items = [draft("numeric", "Berhenti pada hitungan 3.", 45, 60, required="3")]
        wrong = {"condition": "angka == 30", "loop": "while True:", "part": 1, "start": 49, "end": 52,
                 "increment": "angka += 1", "uses_input": False}
        self.assertEqual(enrich(items, {"actions": [wrong]}, []), items)
        result = enrich(items, {"actions": [{**wrong, "condition": "angka == 3"}]},
                        [{"part": 1, "code": "if angka == 3:", "quote": "Berhenti pada hitungan tiga."}])
        self.assertEqual(result[0]["required"], "angka == 3")
        self.assertEqual(result[0]["quote"], "Berhenti pada hitungan tiga.")
        self.assertNotIn("quote", items[0])

    def test_enrichment_uses_verified_prompt_for_the_same_input_variable(self):
        items = [draft("input", "Baca batas dari masukan.", 45, 60, required="input")]
        state = {"parts": [{"start": 0, "end": 60}],
                 "actions": [{"condition": "angka == batas", "loop": "while True:", "part": 1,
                              "start": 50, "end": 52, "increment": "", "uses_input": True}]}
        evidence = {"part": 1, "code": 'batas = int(input("Hitung sampai "))', "quote": "Masukkan batas hitungan."}
        result = enrich(items, state, [evidence])[0]
        self.assertIn(evidence["code"], result["text"])
        self.assertEqual(result["quote"], evidence["quote"])
        self.assertEqual(result["required"], "angka == batas")
        unrelated = {**evidence, "code": evidence["code"].replace("batas", "nilai", 1)}
        self.assertNotIn(unrelated["code"], enrich(items, state, [unrelated])[0]["text"])

    def test_numeric_step_uses_final_count_and_precedes_the_comparison_run(self):
        cues = [
            Segment(1, 3, "Pilih continue atau break pada perulangan."),
            Segment(8, 10, "Cari nilai yang cocok lalu pakai break."),
            Segment(18, 20, "Angka sekarang 2."),
            Segment(22, 24, "Di bawah kondisi ada break."),
            Segment(35, 37, "Angka sekarang 3, nice, cukup finish."),
            Segment(40, 42, "Dia langsung break."),
            Segment(44, 46, "Komentari break."),
            Segment(49, 51, "Semua dijalankan sampai 5."),
        ]
        steps = speech_drafts(video_parts(cues, 55, 55))
        numeric = next(row for row in steps if row["key"] == "numeric")
        contrast = next(row for row in steps if row["key"] == "contrast")
        self.assertEqual(numeric["required"], "3")
        self.assertLess(numeric["start"], contrast["start"])

    def test_exact_details_are_scored_separately_from_keyword_coverage(self):
        reference = {"facts": [{"start": 0, "end": 10, "summary": "Gunakan break.", "terms": [["break"]]}],
                     "code_details": ["angka == 3", "int(input("]}
        rows = [{"start": 1, "end": 4, "summary": "Gunakan break dengan angka == 30 dan int(input(...))."}]
        scores = evaluate(rows, reference)
        self.assertEqual(scores["coverage"], 1)
        self.assertEqual(scores["detail_recall"], 0.5)
