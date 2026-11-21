import unittest

from src.segmentation import Segment, group_segments
from src.summarization import summary_preserves_details


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


class SummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
