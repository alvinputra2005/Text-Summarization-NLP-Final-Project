import unittest
from unittest.mock import patch

from scripts.prepare_video_experiment import clip_text, natural_rows


class VideoExperimentTests(unittest.TestCase):
    def test_clip_text_keeps_cues_from_the_reference_window(self):
        cues = [(0, 3, "Pembuka"), (2, 7, "[Musik] Gunakan append."), (7, 11, "Hasilnya ada di akhir.")]
        self.assertEqual(clip_text(cues, 4, 8), "Gunakan append.")

    def test_rejects_a_whole_video_reused_on_another_split(self):
        video = {"video_id": "JUE9vPXefGI", "topic": "range", "clips": [[1, 10, "Gunakan range.", ["range"]]]}
        manifest = {"train": [video], "validation": [video], "test": []}
        with patch("scripts.prepare_video_experiment.captions", return_value=[(2, 5, "Gunakan range.")]), \
             self.assertRaisesRegex(ValueError, "Video muncul lebih dari sekali"):
            natural_rows(manifest, None)
