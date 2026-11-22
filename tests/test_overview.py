import unittest
from unittest.mock import Mock, patch

from src.overview import (
    BINARY_EXAMPLE, BOOLEAN_EVIDENCE, INPUT_EVIDENCE, NUMBER_EVIDENCE, PROMPT_EXAMPLE,
    STRING_EVIDENCE, TEXT_BOOLEAN_EXAMPLE, find_input_examples, overview_context,
    prepare_evidence, summarize_overview,
)
from src.segmentation import Segment


class OverviewTests(unittest.TestCase):
    def test_evidence_includes_information_across_the_video_without_greeting(self):
        evidence = prepare_evidence([
            Segment(0, 4, "Assalamualaikum teman-teman semua, selamat datang di episode ini."),
            Segment(58, 64, "Gunakan input untuk membaca masukan pengguna."),
            Segment(80, 86, "Data yang dimasukkan pasti string."),
            Segment(180, 186, "Kita casting ke integer baru di casting ke boolean."),
        ])
        self.assertEqual(evidence, (
            "Gunakan input untuk membaca masukan pengguna.",
            "Masukan pengguna dibaca sebagai string.",
            "Untuk memperoleh nilai boolean, konversikan masukan ke integer terlebih dahulu, lalu ke boolean.",
        ))

    def test_summary_uses_evidence_if_model_introduces_an_unsupported_topic(self):
        evidence = ("Gunakan input untuk membaca masukan pengguna.",)
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": Mock(shape=(1, 12))}
        tokenizer.decode.return_value = "Gunakan input untuk membaca masukan pengguna dan konversikan ke float."
        model = Mock()
        model.config.max_position_embeddings = 1024
        with patch("src.summarization.generate_mbart", return_value=[[1]]):
            result = summarize_overview(evidence, tokenizer, model)
        self.assertTrue(result.is_excerpt)
        self.assertEqual(result.text, overview_context(evidence))

    def test_python_input_overview_combines_type_concepts_without_procedure(self):
        evidence = (INPUT_EVIDENCE, STRING_EVIDENCE, NUMBER_EVIDENCE, BOOLEAN_EVIDENCE)
        context = overview_context(evidence)
        self.assertEqual(context.count("."), 2)
        self.assertIn("Masukan berupa string secara bawaan", context)
        self.assertIn("integer atau float dan nilai boolean", context)
        self.assertNotIn("terlebih dahulu", context)
        self.assertNotIn("Gunakan", context)

        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": Mock(shape=(1, 12))}
        tokenizer.decode.return_value = " ".join(evidence)
        model = Mock()
        model.config.max_position_embeddings = 1024
        with patch("src.summarization.generate_mbart", return_value=[[1]]):
            result = summarize_overview(evidence, tokenizer, model)
        self.assertTrue(result.is_excerpt)
        self.assertEqual(result.text, context)

    def test_other_tutorials_group_actions_by_topic_and_keep_cautions(self):
        context = overview_context(("Baca data dari CSV.", "Kemudian, jangan hapus baris yang kosong."))
        self.assertEqual(context, "Video ini membahas pembacaan data dari CSV dan larangan penghapusan baris yang kosong.")

    def test_examples_are_recovered_from_caption_fragments(self):
        examples = find_input_examples([
            Segment(0, 2, "Kita bikin input pakai kurung dan yang di sini teksnya apa."),
            Segment(2, 3, "Biasanya teks pertanyaan."),
            Segment(12, 14, "Kalau kita masukin input di sini 42 misalkan ya."),
            Segment(14, 15, "Data yang keluar adalah STR."),
            Segment(31, 35, "Nilai boolean true. Kalau ini menjadi false tidak menjadi false, tetap true."),
            Segment(47, 50, "Satu jadi true, ya. Kalau ini nol jadi false."),
        ])
        number_example = "Angka 42 yang diketik tetap terbaca sebagai string."
        self.assertEqual(examples, (PROMPT_EXAMPLE, number_example, TEXT_BOOLEAN_EXAMPLE, BINARY_EXAMPLE))
        self.assertEqual(find_input_examples([Segment(0, 1, "Bahasan boolean, integer, dan string.")]), ())

        evidence = (INPUT_EVIDENCE, STRING_EVIDENCE, NUMBER_EVIDENCE, BOOLEAN_EVIDENCE, *examples)
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": Mock(shape=(1, 48))}
        tokenizer.decode.return_value = "Video ini membahas input() dan string, integer, float, dan boolean."
        model = Mock()
        model.config.max_position_embeddings = 1024
        with patch("src.summarization.generate_mbart", return_value=[[1]]):
            result = summarize_overview(evidence, tokenizer, model)
        self.assertTrue(result.is_excerpt)
        self.assertIn(number_example, result.text)
        self.assertIn("\n\n" + TEXT_BOOLEAN_EXAMPLE, result.text)
        self.assertIn(BINARY_EXAMPLE, result.text)
