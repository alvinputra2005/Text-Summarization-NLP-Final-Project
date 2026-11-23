import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from src.gemini_preprocessing import GeminiConfig, gemini_config, prepare_gemini_sources
from src.pipeline import finish_guide
from src.segmentation import Segment
from src.summarization import Summary


CONFIG = GeminiConfig("test-key")
CUES = [
    Segment(2, 5, "Selamat datang di tutorial."),
    Segment(10, 15, "Nah, kita ngambil angka 0 sampai 4 dengan range(5)."),
    Segment(16, 21, "Angka 5 tidak muncul dalam hasil perulangan."),
]
EXCERPT = {
    "start_id": 1, "end_id": 3, "quote": CUES[1].text,
    "cleaned_text": "kita mengambil angka 0 sampai 4 dengan range(5).",
}


def response_with(*excerpts):
    return io.BytesIO(json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": json.dumps({"excerpts": excerpts})},
    ]}}]}).encode())


class GeminiPreprocessingTest(unittest.TestCase):
    def test_accepts_ai_studio_authorization_key(self):
        key = "AQ.example_test-key"
        with patch.dict(os.environ, {"GEMINI_API_KEY": key}, clear=True), patch(
            "src.gemini_preprocessing.dotenv_values", return_value={}
        ):
            config = gemini_config()
        self.assertEqual(config, GeminiConfig(key))
        self.assertNotIn(key, repr(config))

    def test_loads_local_env_and_can_be_disabled(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env = Path(directory) / ".env"
            with patch("src.gemini_preprocessing.ENV_FILE", env):
                self.assertIsNone(gemini_config())
                env.write_text("GEMINI_API_KEY=test-key\nGEMINI_MODEL=gemini-local-test\n", encoding="utf-8")
                self.assertEqual(gemini_config(), GeminiConfig("test-key", "gemini-local-test"))
                self.assertNotIn("test-key", repr(gemini_config()))
                with patch.dict(os.environ, {"GEMINI_ENABLED": "false"}):
                    self.assertIsNone(gemini_config())

    @patch("src.gemini_preprocessing.build_opener")
    def test_only_verified_excerpt_reaches_mbart_with_source_timestamp(self, opener):
        opener.return_value.open.return_value = response_with(EXCERPT)
        with patch("src.pipeline.gemini_config", return_value=CONFIG), patch(
            "src.pipeline.load_cached_mbart", return_value=(object(), object()),
        ), patch("src.summarization.summarize", side_effect=lambda text, *args, **kwargs: Summary(text)) as summarize:
            guide = finish_guide(b"video", CUES, perf_counter(), 0, Mock(), overview=False)

        request = opener.return_value.open.call_args.args[0]
        data = json.loads(request.data)
        self.assertEqual(request.full_url, f"https://generativelanguage.googleapis.com/v1beta/models/{CONFIG.model}:generateContent")
        self.assertEqual(request.get_header("X-goog-api-key"), CONFIG.api_key)
        self.assertEqual(json.loads(data["contents"][0]["parts"][0]["text"]), [
            {"id": index, "text": cue.text} for index, cue in enumerate(CUES, 1)
        ])
        self.assertEqual(summarize.call_args_list[0].args[0], EXCERPT["cleaned_text"])
        self.assertEqual(guide.transcript, tuple(CUES))
        self.assertEqual(guide.steps[0].start, CUES[1].start)
        self.assertEqual(guide.steps[0].end, CUES[1].end)
        self.assertEqual(guide.steps[0].transcript, CUES[1].transcript)
        self.assertEqual(guide.preprocessing, f"Gemini · {CONFIG.model}")

    @patch("src.gemini_preprocessing.build_opener")
    def test_rejects_invented_quote_ids_terms_numbers_negation_and_code_changes(self, opener):
        code = Segment(24, 30, 'Simpan variabel x = [1, 2].')
        opener.return_value.open.return_value = response_with(
            {**EXCERPT, "end_id": 99},
            {**EXCERPT, "quote": "Buat perulangan dengan while."},
            {**EXCERPT, "cleaned_text": "kita mengambil angka 0 sampai 5 dengan range(5)."},
            {**EXCERPT, "cleaned_text": "kita mengambil angka 0 sampai 4 dengan list(5)."},
            {"start_id": 3, "end_id": 3, "quote": CUES[2].text,
             "cleaned_text": "Angka 5 muncul dalam hasil perulangan."},
            {"start_id": 4, "end_id": 4, "quote": code.text,
             "cleaned_text": "Simpan variabel x = [1 2]."},
        )
        self.assertEqual(prepare_gemini_sources(CUES + [code], CONFIG), [])

    @patch("src.pipeline.summarize_segments", return_value=([], "test-mbart", None))
    @patch("src.gemini_preprocessing.build_opener")
    def test_keeps_local_preprocessing_when_api_fails_or_key_is_missing(self, opener, summarize):
        opener.return_value.open.side_effect = HTTPError("https://example.org", 429, "secret detail", {}, io.BytesIO())
        progress = Mock()
        with patch("src.pipeline.gemini_config", return_value=CONFIG):
            guide = finish_guide(b"video", CUES, perf_counter(), 0, progress, overview=False)
        self.assertEqual(guide.preprocessing, "Seleksi teknis lokal")
        self.assertTrue(guide.gemini_quota_exhausted)
        self.assertIn("HTTP 429", guide.preprocessing_note)
        self.assertNotIn("secret detail", guide.preprocessing_note)
        summarize.assert_called_once_with(CUES, progress)
        opener.reset_mock()
        with patch("src.pipeline.gemini_config", return_value=None):
            guide = finish_guide(b"video", CUES, perf_counter(), 0, progress, overview=False)
        self.assertEqual(guide.preprocessing, "Seleksi teknis lokal")
        self.assertFalse(guide.gemini_quota_exhausted)
        self.assertFalse(guide.preprocessing_note)
        opener.assert_not_called()
