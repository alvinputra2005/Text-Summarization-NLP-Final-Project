# Python Tutorial Summarizer

Aplikasi Streamlit untuk mengubah video tutorial Python berbahasa Indonesia menjadi panduan praktik interaktif. Pengguna mengunggah MP4, melihat ringkasan per langkah, lalu memilih penanda waktu untuk membuka bagian demonstrasi di video. Uji awal berfokus pada tutorial Python dan pandas untuk pengolahan CSV.

## Pipeline

1. **Validasi MP4** menggunakan FFprobe untuk memeriksa jalur video dan audio.
2. **Transkripsi audio** menggunakan FFmpeg dan Whisper. Setiap potongan transkrip membawa waktu mulai dan selesai dalam detik.
3. **Segmentasi transkrip** menggunakan Python. Potongan ucapan yang membahas satu langkah dikelompokkan sambil mempertahankan istilah Python dan penanda waktu sumber.
4. **Peringkasan abstraktif** menggunakan **mBART-large-50** melalui Transformers dan PyTorch. Setiap segmen teks menghasilkan instruksi ringkas berbahasa Indonesia.
5. **Navigasi video** menggunakan Streamlit. Setiap instruksi ditautkan ke waktu mulai segmen sumber, dengan cuplikan transkrip untuk pemeriksaan.

## Prasyarat

- Python 3.11 dan Git
- FFmpeg dan FFprobe tersedia di `PATH`
- Koneksi internet untuk pemasangan paket dan unduhan model pertama
- CPU untuk uji pemuatan awal. Gunakan GPU yang tersedia untuk fine-tuning atau jika waktu inferensi CPU tidak sesuai kebutuhan demo

Jalankan berikut dari folder utama proyek di **Windows PowerShell**:

```powershell
py -3.11 --version
git --version
ffmpeg -version
ffprobe -version
```

Jika FFmpeg dan FFprobe belum tersedia, pasang keduanya, lalu buka ulang terminal:

```powershell
winget install --exact --id Gyan.FFmpeg
```

## Setup ulang proyek yang sudah ada

Gunakan folder repo yang sedang dikerjakan. Struktur berikut mengikuti panduan setup yang kamu kirim. Sesuaikan nama modul dengan file yang memang ada di repo.

```text
python-tutorial-summarizer/
├── app.py
├── src/
│   ├── media.py
│   ├── transcription.py
│   ├── segmentation.py
│   └── summarization.py
├── scripts/
│   └── check_mbart.py
├── notebooks/
├── requirements.txt
├── requirements-training.txt
├── .gitignore
├── README.md
└── SETUP.md
```

### Dependensi aplikasi

Paket inti yang diperlukan di `requirements.txt`:

```text
streamlit
openai-whisper
torch>=2.6
transformers
sentencepiece
```

Jika repo sudah mempunyai paket lain yang masih dipakai, pertahankan paket tersebut. Pertahankan versi yang sudah dikunci jika kompatibel dengan pemuatan mBART-large-50.

### Dependensi eksperimen

Jika menjalankan pelatihan atau evaluasi, gunakan `requirements-training.txt`:

```text
-r requirements.txt
datasets
accelerate
jiwer
rouge-score
```

### Instalasi di lingkungan lokal

```powershell
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}
$projectPython = ".\.venv\Scripts\python.exe"
& $projectPython -m pip install -r requirements.txt
& $projectPython -m pip check
& $projectPython -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Jika perlu menjalankan eksperimen:

```powershell
& $projectPython -m pip install -r requirements-training.txt
```

## Konfigurasi mBART-large-50

| Pengaturan | Nilai |
|---|---|
| Checkpoint dasar | `facebook/mbart-large-50` |
| Tokenizer | `MBart50TokenizerFast` |
| Bahasa sumber dan target | `id_ID` |
| Token awal bahasa target saat generasi | `forced_bos_token_id` dari token `id_ID` |
| Konfigurasi lokasi checkpoint | Variabel lingkungan `MBART_MODEL_ID` |
| Contoh checkpoint lokal | `checkpoints/mbart-python` |

Checkpoint dasar dapat dipakai untuk uji pemuatan dan generasi. Gunakan checkpoint hasil fine-tuning untuk mengevaluasi kualitas peringkasan. Muat tokenizer dan model dari checkpoint yang sama. Saat mengekspor hasil fine-tuning, simpan keduanya dalam folder checkpoint.

Untuk memilih checkpoint hasil fine-tuning dari PowerShell:

```powershell
$env:MBART_MODEL_ID = ".\checkpoints\mbart-python"
```

Variabel ini berlaku untuk proses yang dijalankan dari terminal tersebut. Gunakan konfigurasi yang sama pada skrip uji, modul peringkasan dan, jika tersedia, pipeline pelatihan atau evaluasi.

## Uji tokenizer dan model

Buat `scripts/check_mbart.py` atau sesuaikan skrip uji di repo dengan contoh berikut:

```python
import os

import torch
from transformers import AutoModelForSeq2SeqLM, MBart50TokenizerFast


def main():
    model_id = os.environ.get("MBART_MODEL_ID", "facebook/mbart-large-50")
    language = "id_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MBart50TokenizerFast.from_pretrained(
        model_id, src_lang=language, tgt_lang=language
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    language_id = tokenizer.convert_tokens_to_ids(language)
    if language_id is None or language_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer tidak mengenali kode bahasa id_ID")

    source_text = (
        "Baca file CSV menggunakan pandas. Setelah data terbaca, gunakan "
        "dropna untuk menghapus baris yang kolom harganya kosong."
    )
    inputs = tokenizer(source_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, forced_bos_token_id=language_id, max_new_tokens=40
        )

    print(f"Checkpoint: {model_id}")
    print(f"Perangkat: {device}")
    print(f"Tokenizer OK: {inputs['input_ids'].shape[-1]} token input")
    print(f"Model OK: {output.shape[-1]} token output")
    print("Contoh keluaran:", tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
```

```powershell
& $projectPython ".\scripts\check_mbart.py"
```

Pesan `Tokenizer OK` dan `Model OK` menunjukkan model dapat dimuat dan menjalankan generasi. Nilai kualitas ringkasan dengan ringkasan referensi dan data uji tutorial.

## Uji audio dan jalankan aplikasi

Simpan MP4 bernarasi pendek sebagai `samples/local/tutorial.mp4`. Jika perlu menguji Whisper secara terpisah:

```powershell
ffprobe -v error -show_entries stream=index,codec_type,codec_name -of json ".\samples\local\tutorial.mp4"
& $projectPython -m whisper ".\samples\local\tutorial.mp4" --model base --language Indonesian --task transcribe --device cpu --fp16 False --output_format json --output_dir outputs
$transcript = Get-Content ".\outputs\tutorial.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$transcript.segments | Select-Object -First 3 start, end, text | Format-Table -Wrap
```

Setelah integrasi pipeline tersedia, jalankan:

```powershell
& $projectPython -m streamlit run app.py
```

- [ ] MP4 bernarasi dapat diunggah dan diputar
- [ ] Whisper menghasilkan teks serta waktu mulai dan selesai
- [ ] Aplikasi menghasilkan daftar ringkasan dari checkpoint yang dikonfigurasi
- [ ] Klik langkah membuka waktu sumber tanpa memanggil model ulang
- [ ] Saat video diganti, panduan mengikuti unggahan yang aktif
- [ ] Berkas rusak, audio hilang, transkrip kosong, atau model gagal dimuat memunculkan pesan yang sesuai

Simpan `.venv/`, `uploads/`, `checkpoints/`, `outputs/`, `samples/local/`, dan rahasia konfigurasi di luar commit Git. Catat checkpoint dan versi dependensi yang dipakai setelah pengujian berhasil.

## Pelatihan dan evaluasi

Gunakan IndoSum untuk eksperimen awal peringkasan bahasa Indonesia. Untuk evaluasi pada tugas utama, siapkan segmen transkrip video tutorial dengan ringkasan referensi dan anotasi waktu demonstrasi. Jika sebagian video digunakan untuk adaptasi domain, pisahkan data latih, validasi, dan uji berdasarkan video utuh.

Laporkan ROUGE-1, ROUGE-2, ROUGE-L, waktu proses, kelengkapan langkah, ketepatan istilah Python, dan ketepatan penanda waktu. Catat apakah inferensi memakai checkpoint dasar, hasil fine-tuning IndoSum, atau checkpoint yang diadaptasi pada data tutorial.

## Referensi teknis

- [Checkpoint mBART-large-50](https://huggingface.co/facebook/mbart-large-50)
- [Dokumentasi mBART di Transformers](https://huggingface.co/docs/transformers/model_doc/mbart)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Unduhan FFmpeg](https://ffmpeg.org/download.html)
