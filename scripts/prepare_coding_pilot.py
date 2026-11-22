"""Build an original Indonesian coding corpus for a small mBART experiment."""

import argparse
import csv
from pathlib import Path


# A scenario and all its variants belong to one split, as a single video would.
SCENARIOS = {
    "train": (
        ("rentang-nol", "range", (0, 4, 1)), ("rentang-langkah", "range", (2, 12, 2)),
        ("batas-lulus", "branch", (75, "lulus", "ulang")), ("batas-suhu", "branch", (30, "panas", "sejuk")),
        ("pangkat-dua", "function", (2,)), ("pangkat-tiga", "function", (3,)),
        ("csv-harga", "frame", ("harga", "dropna")), ("csv-kota", "frame", ("kota", "fillna")),
        ("pilihan-gabungan", "boolean", ("or",)), ("pilihan-keduanya", "boolean", ("and",)),
    ),
    "validation": (
        ("rentang-batas", "range", (4, 11, 3)), ("batas-umur", "branch", (18, "dewasa", "remaja")),
        ("pangkat-empat", "function", (4,)), ("csv-stok", "frame", ("stok", "fillna")),
    ),
    "test": (
        ("rentang-mundur", "range", (13, 2, -3)), ("batas-biaya", "branch", (90, "tinggi", "rendah")),
        ("pangkat-lima", "function", (5,)), ("csv-email", "frame", ("email", "dropna")),
        ("pilihan-eksklusif", "boolean", ("^",)),
    ),
}


def example(kind, settings, index):
    if kind == "range":
        start, stop, step = settings
        start += index
        stop += index
        values = list(range(start, stop, step))
        call = f"range({start}, {stop}, {step})"
        spoken = f"Perhatikan {call} di perulangan ini. Hasilnya {values}. Angka {stop} menjadi batas yang tidak ikut diambil."
        summary = f"Gunakan {call} untuk menghasilkan {values}. Batas {stop} tidak termasuk dalam hasil."
    elif kind == "branch":
        limit, success, failure = settings
        value = limit + index - 3
        result = success if value >= limit else failure
        spoken = f"Pada contoh if, kita bandingkan nilai {value} dengan batas {limit} menggunakan >=. Program mencetak {result}."
        summary = f"Bandingkan {value} dengan {limit} memakai >= pada if. Hasil percabangannya adalah {result}."
    elif kind == "function":
        power, = settings
        value = index + 2
        result = value ** power
        call = f"pangkat({value})"
        spoken = f"Kita ingin hasil fungsi dapat digunakan lagi, jadi pakai return. Saat {call} menghitung pangkat {power}, nilainya {result}."
        summary = f"Gunakan return agar {call} mengembalikan {result} untuk pangkat {power}."
    elif kind == "frame":
        column, operation = settings
        if operation == "dropna":
            spoken = f"Di sini ada {index + 1} nilai kosong di kolom {column}. Saya pakai dropna() dengan subset kolom {column}, jadi yang dibuang hanya baris dengan {column} kosong."
            summary = f"Hapus {index + 1} baris dengan {column} kosong memakai dropna() dengan subset kolom {column}."
        else:
            spoken = f"Kolom {column} ada isinya yang kosong. Dengan fillna({index + 1}) pada kolom ini, nilai kosong diisi tanpa menghapus baris."
            summary = f"Isi nilai kosong pada kolom {column} dengan fillna({index + 1}) tanpa menghapus baris."
    else:
        operation, = settings
        left, right = bool(index & 1), bool(index & 2)
        result = {"or": left or right, "and": left and right, "^": left ^ right}[operation]
        expression = f"{left} {operation} {right}"
        spoken = f"Untuk contoh Boolean ini kita jalankan {expression}. Saat hasilnya ditampilkan, program mengeluarkan {result}."
        summary = f"Evaluasi {expression}. Hasil operasinya adalah {result}."
    introductions = ("Mari kita lihat percobaan ini.", "Perhatikan apa yang terjadi saat kode dijalankan.", "Kita lihat lagi cara kerjanya.")
    return f"{introductions[index % len(introductions)]} {spoken}", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/coding_pilot"))
    destination = parser.parse_args().output_dir
    destination.mkdir(parents=True, exist_ok=True)
    for split, scenarios in SCENARIOS.items():
        path = destination / f"{split}.csv"
        unique = set()
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "scenario_id", "topic", "text", "summary", "source"))
            for scenario, topic, settings in scenarios:
                for index in range(4 if topic == "boolean" else 6):
                    text, summary = example(topic, settings, index)
                    if (text, summary) in unique:
                        continue
                    unique.add((text, summary))
                    writer.writerow((f"{scenario}-{index + 1}", scenario, topic, text, summary, "original_synthetic"))
        print(f"{path}: {len(unique)} examples")


if __name__ == "__main__":
    main()
