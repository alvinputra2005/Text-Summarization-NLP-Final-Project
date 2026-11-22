from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.pipeline import Guide, YouTubeInspection
from src.segmentation import Segment


class YouTubeConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.link = "https://youtu.be/M7lc1UVf-VE"
        self.inspection = YouTubeInspection("https://www.youtube.com/watch?v=M7lc1UVf-VE", None, 0.3)

    def check_link(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        app.text_input(key="youtube-link").set_value(self.link).run()
        app.button(key="make-youtube-guide").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Subtitle bahasa Indonesia belum ditemukan" in warning.value for warning in app.warning))
        return app

    def test_existing_session_guide_without_preprocessing_fields_still_renders(self):
        fields = vars(Guide(self.inspection.video, [], "test-mbart", status="unclear")).copy()
        fields.pop("preprocessing")
        fields.pop("preprocessing_note")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.session_state["guide"] = SimpleNamespace(**fields, summary_points=())
        app.session_state["video_name"] = "YouTube"
        app.session_state["seek_start"] = None
        app.session_state["seek_revision"] = 0
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Tidak tercatat" in item.value for item in app.get("html")))

    def test_missing_subtitles_await_approval_and_cancel_skips_audio(self):
        with patch("src.pipeline.inspect_youtube", return_value=self.inspection), patch("src.pipeline.build_youtube_guide") as build:
            app = self.check_link()
            next(button for button in app.button if button.label == "Batal").click().run()
            self.assertTrue(any("Pemrosesan dibatalkan" in item.value for item in app.info))
            self.assertFalse(app.warning)
            build.assert_not_called()

    def test_approval_processes_the_inspected_video_once(self):
        guide = Guide(self.inspection.video, [], "test-mbart", status="no_speech")
        with patch("src.pipeline.inspect_youtube", return_value=self.inspection) as inspect, patch(
            "src.pipeline.build_youtube_guide", return_value=guide,
        ) as build:
            app = self.check_link()
            build.assert_not_called()
            next(button for button in app.button if button.label == "Lanjut dengan Whisper").click().run()
            self.assertFalse(app.exception)
            inspect.assert_called_once_with(self.link)
            self.assertEqual(build.call_args.args[0], self.inspection)
            self.assertEqual(build.call_count, 1)

    def test_editing_the_link_clears_the_old_confirmation(self):
        with patch("src.pipeline.inspect_youtube", return_value=self.inspection), patch("src.pipeline.build_youtube_guide") as build:
            app = self.check_link()
            app.text_input(key="youtube-link").set_value("https://youtu.be/JUE9vPXefGI").run()
            self.assertFalse(app.warning)
            build.assert_not_called()

    def test_available_subtitles_skip_whisper_confirmation(self):
        source = YouTubeInspection(self.inspection.video, ([Segment(1, 5, "Gunakan range(5).")], "Subtitle Indonesia YouTube"), 0.3)
        guide = Guide(source.video, [], "test-mbart", status="no_speech")
        with patch("src.pipeline.inspect_youtube", return_value=source), patch(
            "src.pipeline.build_youtube_guide", return_value=guide,
        ) as build:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            app.text_input(key="youtube-link").set_value(self.link).run()
            app.button(key="make-youtube-guide").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.warning)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(build.call_args.args[0], source)
