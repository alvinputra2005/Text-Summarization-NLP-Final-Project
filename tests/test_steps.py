import unittest
from unittest.mock import Mock, patch

import torch
from transformers import BatchEncoding

from src.pipeline import Guide, Step
from src.segmentation import Segment, group_segments
from src.step_sources import prepare_step_sources
from src.summarization import generate_mbart, model_reference, summarize, summary_preserves_details


class SegmentationTests(unittest.TestCase):
    def test_related_segments_share_source_span(self):
        groups = group_segments([
            Segment(0.2, 2.4, "  Baca data CSV dengan pandas."),
            Segment(2.4, 4.8, " Gunakan berkas penjualan."),
            Segment(5.0, 8.8, " Hapus baris dengan harga kosong."),
        ])
        self.assertEqual(groups, [
            Segment(0.2, 4.8, "  Baca data CSV dengan pandas.\n Gunakan berkas penjualan."),
            Segment(5.0, 8.8, " Hapus baris dengan harga kosong."),
        ])

    def test_break_for_silence_and_skip_empty_segments(self):
        first = Segment(0.0, 1.0, "Periksa isi data.")
        second = Segment(5.0, 6.8, "Periksa data penjualan lagi.")
        self.assertEqual(group_segments([first, Segment(1.0, 2.0, "  "), second]), [first, second])


class GuideTests(unittest.TestCase):
    def test_search_matches_summary_and_transcript_and_keeps_original_index(self):
        read = Step(0, 4, "Baca CSV.", "Muat data.", False)
        clean = Step(4, 8, "Gunakan dropna.", "Hapus harga kosong.", False)
        guide = Guide(b"", [read, clean], "example")

        self.assertEqual(guide.find_steps("DROPNA harga"), [(1, clean)])
        self.assertEqual(guide.find_steps("dropna CSV"), [])
        self.assertEqual(guide.find_steps(" "), [(0, read), (1, clean)])


class StepSourceTests(unittest.TestCase):
    def test_demonstrations_keep_transcript_times_and_test_values(self):
        segments = [
            Segment(0, 4, "Assalamualaikum teman-teman, kembali lagi di tutorial."),
            Segment(10, 14, "Data sama dengan, kita bikin input pakai kurung."),
            Segment(14, 19, "Biasanya teks pertanyaan. Misalkan kita tulis masukkan data."),
            Segment(30, 35, "Kalau kita masukin input di sini 42 misalkan ya."),
            Segment(35, 39, "Data yang keluar adalah STR."),
            Segment(50, 55, "Berarti ini harus kita casting dulu ke integer."),
            Segment(55, 60, "Baru casting ke boolean. Satu jadi true, ya. Kalau ini nol jadi false."),
        ]
        prompt, example, binary = prepare_step_sources(segments)
        self.assertEqual([(item.segment.start, item.segment.end) for item in (prompt, example, binary)], [
            (10, 19), (30, 39), (50, 60),
        ])
        self.assertIn("42", example.text)
        self.assertIn("Data yang keluar adalah STR.", example.segment.transcript)
        self.assertIn("0 menjadi False dan 1 menjadi True", binary.text)

    def test_pandas_actions_are_still_separate(self):
        steps = prepare_step_sources([
            Segment(2, 5, "Baca CSV dengan pandas."),
            Segment(5, 8, "Hapus baris dengan harga kosong."),
        ])
        self.assertEqual([step.text for step in steps], ["Baca CSV dengan pandas.", "Hapus baris dengan harga kosong."])

    def test_dropna_recovers_a_function_split_by_asr(self):
        [step] = prepare_step_sources([
            Segment(10, 16, "Selanjutnya hapus baris dengan hanka kosong gunakan fungsi drop na dan pilih olom harga."),
        ])
        self.assertEqual(step.text, "Hapus baris yang harganya kosong dengan dropna() pada kolom harga.")
        [caution] = prepare_step_sources([
            Segment(10, 16, "Jangan hapus baris yang kosong dengan dropna lalu pilih kolom harga."),
        ])
        self.assertTrue(caution.text.startswith("Jangan hapus"))


class SummaryTests(unittest.TestCase):
    def test_model_reference_uses_default_and_accepts_checkpoint_override(self):
        for value, expected in [
            (" ", "facebook/mbart-large-50"),
            (" ./checkpoints/mbart-python ", "./checkpoints/mbart-python"),
            ("tim/mbart-python", "tim/mbart-python"),
        ]:
            with self.subTest(value=value), patch.dict("os.environ", MBART_MODEL_ID=value):
                self.assertEqual(model_reference(), expected)

    def test_generation_uses_eos_and_forces_indonesian(self):
        language_id = 250032
        tokenizer = Mock(lang_code_to_id={"id_ID": language_id}, eos_token_id=2)
        model = Mock(device=torch.device("cpu"))
        inputs = BatchEncoding({
            "input_ids": torch.tensor([[language_id, 412, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

        self.assertIs(generate_mbart(inputs, tokenizer, model), model.generate.return_value)
        options = model.generate.call_args.kwargs
        self.assertEqual(options["decoder_start_token_id"], 2)
        self.assertEqual(options["forced_bos_token_id"], language_id)
        self.assertTrue(torch.equal(options["attention_mask"], inputs["attention_mask"]))

    def test_preserves_function_argument_case_and_quantity(self):
        source = 'Untuk tahap ini, pakai dropna(subset=["Harga"]) pada lima baris penjualan.'
        concise = 'Pakai dropna(subset=["Harga"]) pada lima baris penjualan.'
        self.assertTrue(summary_preserves_details(source, concise))
        self.assertFalse(summary_preserves_details(source, concise.replace('"Harga"', '"harga"')))
        self.assertFalse(summary_preserves_details(source, concise.replace('lima', 'enam')))
        self.assertFalse(summary_preserves_details(source, concise.replace('dropna', 'fillna')))

    def test_preserves_scope_of_negation(self):
        source = "Jangan hapus baris kosong. Simpan hasil ke CSV."
        self.assertTrue(summary_preserves_details(source, source))
        self.assertFalse(summary_preserves_details(source, "Hapus baris kosong. Jangan simpan hasil ke CSV."))

    def test_step_preserves_the_boolean_result_and_conversion(self):
        source = "Masukan diproses dengan int() dan bool(). Hasilnya 0 menjadi False dan 1 menjadi True."
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": Mock(shape=(1, 30))}
        tokenizer.decode.return_value = "Masukan diproses dengan int() dan bool(). Hasilnya 0 menjadi True dan 1 menjadi False."
        model = Mock()
        model.config.max_position_embeddings = 1024
        with patch("src.summarization.generate_mbart", return_value=[[1]]):
            result = summarize(source, tokenizer, model, required_clause="0 menjadi False dan 1 menjadi True")
        self.assertTrue(result.is_excerpt)
        self.assertEqual(result.text, source)


if __name__ == "__main__":
    unittest.main()
