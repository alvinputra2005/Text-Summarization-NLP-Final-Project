from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.media import MediaError
from src.pipeline import Step, YouTubeInspection, build_youtube_guide, inspect_youtube
from src.segmentation import Segment
from src.youtube import download_youtube_audio, parse_youtube_captions, youtube_watch_url


class YouTubeTests(unittest.TestCase):
    def test_preflight_stops_after_subtitle_lookup(self):
        with patch("src.pipeline.fetch_youtube_captions", return_value=None), patch("src.pipeline.download_youtube_audio") as download:
            inspected = inspect_youtube("https://youtu.be/M7lc1UVf-VE")
        self.assertIsNone(inspected.captions)
        self.assertEqual(inspected.video, "https://www.youtube.com/watch?v=M7lc1UVf-VE")
        download.assert_not_called()

    def test_a_checked_link_does_not_repeat_subtitle_lookup_when_approved(self):
        checked = YouTubeInspection("https://www.youtube.com/watch?v=M7lc1UVf-VE", None, 1)
        with patch("src.pipeline.fetch_youtube_captions") as fetch, patch("src.pipeline.download_youtube_audio") as download, patch(
            "src.pipeline.transcribe", return_value=[],
        ) as transcribe, patch("src.pipeline.finish_guide"):
            build_youtube_guide(checked, Mock(), prefer_captions=True)
        fetch.assert_not_called()
        transcribe.assert_called_once_with(download.return_value)

    def test_normalizes_supported_video_links_and_ignores_input_timestamp(self):
        video_id = "M7lc1UVf-VE"
        links = (
            f"https://www.youtube.com/watch?list=playlist&v={video_id}&t=20",
            f"youtu.be/{video_id}?si=abc",
            f"https://m.youtube.com/shorts/{video_id}/",
            f"https://www.youtube.com/live/{video_id}",
            f"https://www.youtube-nocookie.com/embed/{video_id}",
        )
        for link in links:
            with self.subTest(link=link):
                self.assertEqual(youtube_watch_url(link), f"https://www.youtube.com/watch?v={video_id}")

    def test_rejects_other_hosts_credentials_and_nonvideo_urls(self):
        links = (
            "https://www.youtube.com.example/watch?v=M7lc1UVf-VE",
            "https://example.com@www.youtube.com/watch?v=M7lc1UVf-VE",
            "https://youtu.be/M7lc1UVf-VE/anything",
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/watch?v=M7lc1UVf-VE&v=aaaaaaaaaaa",
        )
        for link in links:
            with self.subTest(link=link), self.assertRaises(MediaError):
                youtube_watch_url(link)

    def test_audio_downloader_accepts_only_a_file_inside_the_temporary_directory(self):
        video = "https://www.youtube.com/watch?v=M7lc1UVf-VE"
        with TemporaryDirectory() as scratch, patch("src.youtube.subprocess.run") as run:
            directory = Path(scratch)
            audio = directory / "source.webm"
            audio.write_bytes(b"audio fixture")
            run.return_value = CompletedProcess([], 0, "source.webm\n", "")
            self.assertEqual(download_youtube_audio(video, directory), audio.resolve())
            self.assertEqual(run.call_args.args[0][-1], video)
            self.assertEqual(run.call_args.kwargs["cwd"], directory)
            run.return_value = CompletedProcess([], 0, "../source.webm\n", "")
            with self.assertRaises(MediaError):
                download_youtube_audio(video, directory)

    def test_download_failure_is_reported_as_media_error(self):
        completed = CompletedProcess([], 1, "", "ERROR: This video is unavailable")
        with patch("src.youtube.subprocess.run", return_value=completed), self.assertLogs("src.youtube"):
            with self.assertRaisesRegex(MediaError, "Audio YouTube tidak bisa diambil"):
                download_youtube_audio("https://www.youtube.com/watch?v=M7lc1UVf-VE", Path("."))

    def test_youtube_guide_transcribes_the_download_and_keeps_whisper_timestamps(self):
        segment = Segment(10, 13, "Gunakan dropna.")
        steps = [Step(segment.start, segment.end, segment.text, "Gunakan dropna.", False)]
        progress = Mock()
        with patch.dict("os.environ", {"VIDEO_OCR_ENABLED": "false"}), patch("src.pipeline.download_youtube_audio") as download, patch(
            "src.pipeline.transcribe", return_value=[segment],
        ) as transcribe, patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.pipeline.summarize_segments", return_value=(steps, "example-mbart", None),
        ) as summarize:
            guide = build_youtube_guide("https://youtu.be/M7lc1UVf-VE?t=20", progress)
        video, scratch = download.call_args.args
        self.assertEqual(video, guide.video)
        self.assertFalse(scratch.exists())
        transcribe.assert_called_once_with(download.return_value)
        summarize.assert_called_once_with([segment], progress)
        self.assertEqual(guide.video, "https://www.youtube.com/watch?v=M7lc1UVf-VE")
        self.assertEqual(guide.steps, steps)
        self.assertEqual(guide.transcription_model, "base")
        progress.assert_called_with(1.0, "Panduan siap.")

    def test_reads_caption_segments_in_seconds_and_ignores_empty_events(self):
        document = {"events": [
            {"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": " "}]},
            {"tStartMs": 15200, "dDurationMs": 2800, "segs": [{"utf8": "Gunakan "}, {"utf8": "input()."}]},
        ]}
        self.assertEqual(parse_youtube_captions(document), [Segment(15.2, 18, "Gunakan input().")])

    def test_overview_prefers_captions_and_skips_audio_and_whisper(self):
        captions = [Segment(10, 15, "Gunakan input().")]
        with patch("src.pipeline.fetch_youtube_captions", return_value=(captions, "Subtitle otomatis YouTube")), patch(
            "src.pipeline.download_youtube_audio",
        ) as download, patch("src.pipeline.transcribe") as transcribe, patch("src.pipeline.finish_guide") as finish:
            build_youtube_guide("https://youtu.be/M7lc1UVf-VE", Mock(), overview=True)
        download.assert_not_called()
        transcribe.assert_not_called()
        self.assertEqual(finish.call_args.args[1], captions)
        self.assertEqual(finish.call_args.kwargs["transcript_source"], "Subtitle otomatis YouTube")

    def test_noisy_captions_retry_audio_and_use_the_more_readable_transcript(self):
        captions = [Segment(i, i + 1, "Oke nah gitu ini anu") for i in range(12)]
        revised = [Segment(0, 5, "Operator not membalik nilai Boolean.")]
        with patch("src.pipeline.fetch_youtube_captions", return_value=(captions, "Subtitle otomatis YouTube")), patch(
            "src.pipeline.download_youtube_audio",
        ) as download, patch("src.pipeline.transcribe", return_value=revised) as transcribe, patch(
            "src.pipeline.finish_guide",
        ) as finish:
            build_youtube_guide("https://youtu.be/M7lc1UVf-VE", Mock(), prefer_captions=True)
        transcribe.assert_called_once_with(download.return_value, retry_noisy=True)
        self.assertEqual(finish.call_args.args[1], revised)
        self.assertIsNone(finish.call_args.kwargs["transcript_source"])

    def test_noisy_captions_are_kept_when_audio_is_unavailable(self):
        captions = [Segment(i, i + 1, "Oke nah gitu ini anu") for i in range(12)]
        with patch("src.pipeline.fetch_youtube_captions", return_value=(captions, "Subtitle otomatis YouTube")), patch(
            "src.pipeline.download_youtube_audio", side_effect=MediaError("Audio gagal."),
        ), patch("src.pipeline.finish_guide") as finish:
            build_youtube_guide("https://youtu.be/M7lc1UVf-VE", Mock(), prefer_captions=True)
        self.assertEqual(finish.call_args.args[1], captions)
        self.assertIn("transkripsi ulang gagal", finish.call_args.kwargs["transcript_note"])
