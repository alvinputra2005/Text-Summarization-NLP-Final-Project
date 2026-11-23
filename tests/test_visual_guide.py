from pathlib import Path
from dataclasses import asdict
from time import perf_counter
from unittest import TestCase
from unittest.mock import Mock, patch

from src.gemini_preprocessing import GEMINI_QUOTA_NOTICE, GeminiConfig
from src.pipeline import Step, YouTubeInspection, build_youtube_guide, finish_guide, retry_sparse_transcript
from src.segmentation import Segment
from src.step_sources import StepSource
from src.summarization import Summary
from src.transcript_quality import coding_transcript_status
from src.video_context import visual_branch_sources, visual_call_sources
from src.video_ocr import OCRLine, find_branches
from src.video_outline import recover_tutorial_sources


CUES = [Segment(10, 13, "Gunakan dropna untuk menghapus baris yang kosong.")]


class VisualGuideTest(TestCase):
    def test_opening_does_not_hide_a_concrete_instruction_in_the_same_cue(self):
        cues = [Segment(5, 12, "Selamat datang, gunakan nama[1:4] untuk mengambil bagian string.")]
        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.pipeline.load_cached_mbart", side_effect=RuntimeError("not enough memory"),
        ):
            guide = finish_guide(b"mp4", cues, perf_counter(), 0, Mock(), overview=False)
        self.assertEqual(guide.status, "ready")
        self.assertEqual(guide.summary_points, ("gunakan nama[1:4] untuk mengambil bagian string.",))
        self.assertIn("mBART tidak dapat dimuat", guide.model)

    def test_unclear_speech_can_use_a_branch_chain_from_screen_without_gemini(self):
        lines = [OCRLine(text, indent, 25 * i, 92) for i, (text, indent) in enumerate([
            ("if jumlah > 8:", 25), ('print("Kirim")', 60), ("else:", 25), ('print("Tunggu")', 60),
        ])]
        branches = [asdict(branch) for branch in find_branches(lines)]
        frames = [{"second": second, "branches": branches} for second in (12, 14)]
        self.assertFalse(visual_branch_sources([], [{"frames": frames[:1]}], [], screen_only=True))
        lone_if = [{**frame, "branches": branches[:1]} for frame in frames]
        self.assertFalse(visual_branch_sources([], [{"frames": lone_if}], [], screen_only=True))
        cues = [Segment(0, 7, "Oke anu kita taruh ini di sini")]
        context = {"parts": [{"id": 1, "frames": frames}], "frame_count": 2, "ocr_seconds": 1}
        with patch("src.video_ocr.tesseract_path", return_value="tesseract"), patch(
            "src.video_context.scan_context", return_value=context,
        ) as scan, patch("src.pipeline.gemini_config") as gemini, patch("src.pipeline.load_cached_mbart") as model:
            guide = finish_guide(b"mp4", cues, perf_counter(), 0, Mock(), overview=False, visual_path=Path("v.mp4"))
        self.assertTrue(scan.call_args.kwargs["screen_only"])
        gemini.assert_not_called()
        model.assert_not_called()
        self.assertEqual(guide.status, "ready")
        self.assertEqual(len(guide.summary_points), 2)
        self.assertIn("jumlah > 8", guide.summary_points[0])

    def test_ocr_uses_an_if_elif_else_chain_confirmed_in_two_frames(self):
        text = [
            ("if stok >= 12:", 50), ('print("Kirim")', 88),
            ("elif stok > 0:", 50), ('print("Tunggu")', 88),
            ("else:", 50), ('print("Habis")', 88),
        ]
        lines = [OCRLine(line, x, 20 * i, 92) for i, (line, x) in enumerate(text)]
        branches = [asdict(branch) for branch in find_branches(lines)]
        self.assertEqual(len(branches), 3)
        self.assertEqual(find_branches(lines[4:]), [])
        self.assertEqual(find_branches([lines[0], OCRLine('print("Kirim")', 50, 20, 92)]), [])
        segments = [Segment(0, 4, "Sekarang kita bahas percabangan if dan else.")]
        frames = [{"second": second, "branches": branches} for second in (9, 13)]
        part = {"id": 1, "frames": frames}
        self.assertEqual(visual_branch_sources(segments, [{**part, "frames": frames[:1]}], []), [])
        result = visual_branch_sources(segments, [part], [])
        self.assertEqual([row.segment.start for row in result], [9, 9, 9])
        self.assertIn("stok >= 12", result[0].text)
        self.assertIn('Jika benar, jalankan `print("Tunggu")`', result[1].text)
        self.assertIn('print("Habis")', result[2].text)
        self.assertIn("stok > 0", result[2].text)

    def test_lesson_announcements_and_environment_mentions_are_not_steps(self):
        segments = [
            Segment(0, 3, "Sekarang topik yang kita bahas adalah if dan else statement."),
            Segment(3, 8, "Kalau misalkan kita dari tutorial sebelumnya, kita memprogram satu alur."),
            Segment(8, 12, "Coba kita run ini adalah Python 3, ini adalah demo.py."),
            Segment(12, 16, "Kita akan coba latihannya penggunaan if else."),
        ]
        with patch("src.pipeline.gemini_config", return_value=None), patch("src.pipeline.load_cached_mbart") as load:
            guide = finish_guide(b"playable", segments, perf_counter(), 0, Mock(), overview=False)
        self.assertFalse(guide.summary_points)
        load.assert_not_called()

    def test_an_unlisted_function_survives_unpunctuated_split_and_model_rejection(self):
        cues = [
            Segment(0, 2, "oke nah kita coba ini gitu"),
            Segment(3, 6, "pakai library.queueTask() untuk menyimpan"),
            Segment(6, 9, "pekerjaan baru jika statusnya tidak batal"),
        ]
        [candidate] = recover_tutorial_sources(cues)
        self.assertEqual((candidate.segment.start, candidate.segment.end), (3, 9))
        self.assertIn("library.queueTask()", candidate.text)
        self.assertIn("tidak batal", candidate.text)
        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.pipeline.load_cached_mbart", return_value=(object(), object()),
        ), patch("src.summarization.summarize", return_value=Summary("ini gitu.")):
            guide = finish_guide(b"mp4", cues, perf_counter(), 0, Mock(), overview=False)
        self.assertEqual(guide.status, "ready")
        self.assertEqual(guide.summary_points, (candidate.text,))
        self.assertTrue(guide.steps[0].is_excerpt)

    def test_ordinary_actions_and_filler_do_not_turn_into_coding_steps(self):
        cues = [Segment(0, 4, "Gunakan pisau untuk memotong sayuran"),
                Segment(4, 9, "nah ini kita coba lagi gitu")]
        self.assertEqual(recover_tutorial_sources(cues), [])

    def test_retry_uses_a_better_audio_pass_only_when_a_step_is_recovered(self):
        weak = [Segment(i * 3, i * 3 + 2, "nah itu anu gitu ya pilih format string") for i in range(6)]
        better = [Segment(5, 10, "Pakai library.queueTask() untuk menyimpan pekerjaan baru.")]
        with patch("src.pipeline.transcribe", side_effect=[better, []]) as transcribe:
            self.assertIs(retry_sparse_transcript(Path("video.mp4"), weak, Mock()), better)
            self.assertIs(retry_sparse_transcript(Path("video.mp4"), weak, Mock()), weak)
            self.assertIs(retry_sparse_transcript(Path("video.mp4"), better, Mock()), better)
        self.assertEqual(transcribe.call_count, 2)
        transcribe.assert_called_with(Path("video.mp4"), retry_noisy=True)

    def test_visual_matching_accepts_an_unlisted_method_named_in_the_transcript(self):
        cues = [Segment(4, 10, "Pakai queueTask untuk menyimpan tugas di kode program.")]
        part = {"id": 1, "start": 0, "end": 20, "frames": [
            {"lines": ["client.queueTask(priority=2)"]}, {"lines": ["client.queueTask(priority=2)"]},
        ]}
        steps, count = visual_call_sources(cues, [part], [])
        self.assertEqual(count, 1)
        self.assertIn("client.queueTask(priority=2)", steps[0].text)
    def test_technical_transcript_falls_back_when_model_produces_no_usable_steps(self):
        cues = [Segment(3, 9, "Pada f-string, letakkan variabel di dalam kurung kurawal untuk menampilkan nilainya.")]
        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.pipeline.summarize_segments", return_value=([], "test-mbart", None),
        ):
            guide = finish_guide(b"mp4", cues, perf_counter(), 0, Mock(), overview=False)
        self.assertEqual(guide.status, "ready")
        self.assertIn("kurung kurawal", guide.summary_points[0])
        self.assertTrue(guide.steps[0].is_excerpt)

    def test_spoken_python_title_and_while_break_can_reach_visual_pipeline(self):
        self.assertEqual(coding_transcript_status([Segment(0, 5, "Di piton kita akan membahas perulangan.")]), "ready")
        self.assertEqual(coding_transcript_status([Segment(0, 5, "Saat di dalam while, jalankan break.")]), "ready")

    def test_visual_quota_continues_with_local_sources_and_skips_gemini_text(self):
        chosen = StepSource(CUES[0], "Gunakan df.dropna() untuk menghapus baris kosong.", "df.dropna()")
        step = Step(10, 13, CUES[0].transcript, chosen.text, False)
        progress = Mock()
        context = {"parts": [], "frame_count": 2, "ocr_seconds": 0.5}
        with patch("src.pipeline.gemini_config", return_value=GeminiConfig("fake")), patch(
            "src.video_ocr.tesseract_path", return_value="tesseract",
        ), patch("src.video_context.scan_context", return_value=context), patch(
            "src.video_context.read_with_gemini",
            return_value={"quota_exhausted": True, "failed": True, "status": "HTTP 429", "verified": [{"unused": True}]},
        ), patch("src.video_context.contextual_sources", return_value=([chosen], 1)) as contextualize, patch(
            "src.pipeline.prepare_gemini_sources",
        ) as send_text, patch(
            "src.pipeline.summarize_segments", return_value=([step], "test-mbart", None),
        ) as summarize:
            guide = finish_guide(b"mp4", CUES, perf_counter(), 0, progress, overview=False, visual_path=Path("video.mp4"))
        contextualize.assert_called_once_with(CUES, context, [])
        send_text.assert_not_called()
        summarize.assert_called_once_with(CUES, progress, sources=[chosen])
        progress.assert_any_call(0.56, GEMINI_QUOTA_NOTICE)
        self.assertTrue(guide.gemini_quota_exhausted)
        self.assertEqual(guide.visual.steps, 1)
        self.assertEqual(guide.summary_points, (chosen.text,))

    def test_tesseract_unavailable_falls_back_to_text(self):
        progress = Mock()
        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.video_ocr.tesseract_path", side_effect=FileNotFoundError,
        ), patch("src.pipeline.summarize_segments", return_value=([], "test-mbart", None)) as summarize:
            guide = finish_guide(b"mp4", CUES, perf_counter(), 0, progress, overview=False, visual_path=Path("video.mp4"))
        summarize.assert_called_once_with(CUES, progress)
        self.assertIn("Pembacaan frame tidak tersedia", guide.visual.note)
        self.assertFalse(guide.gemini_quota_exhausted)

    def test_youtube_ocr_reads_frames_before_the_temporary_download_is_removed(self):
        inspection = YouTubeInspection("https://www.youtube.com/watch?v=M7lc1UVf-VE", (CUES, "Subtitle"), 0)
        observed = []

        def download(_url, folder):
            path = folder / "source.mp4"
            path.touch()
            return path

        def scan(path, segments, *_):
            self.assertTrue(path.is_file())
            self.assertEqual(segments, CUES)
            observed.append(path)
            return {"parts": [], "frame_count": 2, "ocr_seconds": 0.1}

        with patch("src.pipeline.gemini_config", return_value=None), patch(
            "src.video_ocr.tesseract_path", return_value="tesseract",
        ), patch("src.video_ocr.download_video", side_effect=download), patch(
            "src.video_context.scan_context", side_effect=scan,
        ), patch("src.video_context.contextual_sources", return_value=([], 0)), patch(
            "src.pipeline.summarize_segments", return_value=([], "test-mbart", None),
        ), patch("src.pipeline.transcribe") as transcribe:
            guide = build_youtube_guide(inspection, Mock())
        transcribe.assert_not_called()
        self.assertEqual(guide.video, inspection.video)
        self.assertEqual(guide.visual.frames, 2)
        self.assertFalse(observed[0].exists())

    def test_call_is_used_only_when_repeated_and_mentioned_in_the_same_section(self):
        line = 'df.dropna(subset=["harga"])'
        first = {"lines": [line]}
        second = {"lines": [line]}
        part = {"id": 1, "start": 0, "end": 30, "frames": [first, second]}
        steps, count = visual_call_sources(CUES, [part], [])
        self.assertEqual(count, 1)
        self.assertEqual(steps[0].required_clause, line)
        self.assertEqual(visual_call_sources(CUES, [{**part, "frames": [first]}], []), ([], 0))
        self.assertEqual(visual_call_sources(CUES, [{**part, "start": 30, "end": 60}], []), ([], 0))
