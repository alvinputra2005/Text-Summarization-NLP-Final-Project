# Panduan Pemrograman

Aplikasi Streamlit lokal yang meringkas tutorial pemrograman berbahasa Indonesia dari YouTube. Tempel tautan video publik, lalu tekan **Buat Panduan**. Pengujian video berfokus pada Python.

Pengantar dan poin utama muncul di samping video. **Analisis pemrosesan** menampilkan alur, waktu proses, transkrip bertanda waktu, dan teks yang dipilih untuk mBART. Evaluasi penelitian ada di [notebook terpadu](notebooks/Pipeline_NLP_Video_Pemrograman.ipynb). Bila narasi kosong atau materinya belum bisa dikenali, pemutar tetap tersedia bersama status pemeriksaan.

## Pipeline

1. Aplikasi memeriksa subtitle Indonesia dan meminta keputusan sebelum menjalankan Whisper jika subtitle tidak ditemukan atau sulit dibaca.
2. Aplikasi merapikan dan memilih cuplikan narasi dari transkrip. Jika tersedia, OCR membaca kode di layar dan Gemini mengaitkan cuplikan narasi dengan bukti visual yang diperiksa terhadap transkrip dan hasil OCR.
3. mBART-50 meringkas teks tiap langkah. Streamlit menampilkan panduan, transkrip bertanda waktu, dan durasi proses.

## Jalankan di Windows PowerShell

Pasang Python 3.11, FFmpeg, dan FFprobe. Untuk **Link YouTube**, siapkan juga Node.js 22 atau lebih baru di PATH serta koneksi internet. Lihat [SETUP.md](SETUP.md) untuk prasyarat dan cara menguji audio. Jalankan dari akar repo.

```powershell
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
}
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
& $projectPython -m pip install -r requirements.txt
& $projectPython -m pip check
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
```

Jika Python 3.11 dipasang melalui `uv`, gunakan `uv venv --python 3.11 --seed .venv` untuk membuat lingkungan. Buka alamat lokal yang ditampilkan Streamlit. Untuk uji awal di CPU, pilih video YouTube pendek. Pengunduhan pertama memerlukan koneksi internet dan ruang cache. Bobot PyTorch model dasar berukuran sekitar 2,44 GB. [Berkas checkpoint mBART](https://huggingface.co/facebook/mbart-large-50/tree/main)

Whisper memakai `faster-whisper` pada CPU INT8 secara bawaan. Untuk membandingkan dengan OpenAI Whisper, jalankan aplikasi atau skrip evaluasi setelah mengatur `$env:WHISPER_BACKEND = 'openai'`. Gunakan `Remove-Item Env:WHISPER_BACKEND` untuk kembali ke pengaturan bawaan.

Atur `WHISPER_MODEL_ID` ke `small` atau folder lokal checkpoint Faster Whisper untuk menguji akurasi yang lebih tinggi. Petunjuk untuk sampel uji NSS-ID dan pilot coding tersedia di [panduan dataset](data/README.md).

Atur `WHISPER_BEAM_SIZE` ke angka 1 sampai 5 untuk mengubah pencarian hasil transkripsi. `scripts.tune_whisper` membandingkan konfigurasi pada split validasi NSS-ID dan menguji pilihan teratas pada penutur di split uji.

Untuk memakai konfigurasi lokal yang terpilih pada pengujian terbaru, jalankan dari akar repo dengan folder checkpoint yang sudah diunduh.

```powershell
$env:WHISPER_BACKEND = 'faster'
$env:WHISPER_MODEL_ID = 'data/raw/models/faster-whisper-turbo'
$env:WHISPER_BEAM_SIZE = '3'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## Konfigurasi model

Default aplikasi adalah `facebook/mbart-large-50`. `MBart50TokenizerFast` menggunakan `id_ID` sebagai bahasa sumber dan target. Generasi dimulai dengan token pembuka dekoder `eos`, lalu memaksa token bahasa target `id_ID`. Checkpoint dasar merupakan model pralatih untuk uji pemuatan dan generasi. Gunakan ringkasan referensi tutorial untuk menilai kualitas peringkasan model yang sudah di-fine-tune. [Kartu model](https://huggingface.co/facebook/mbart-large-50) · [Dokumentasi Transformers](https://huggingface.co/docs/transformers/model_doc/mbart)

Atur `MBART_MODEL_ID` sebelum menjalankan aplikasi atau skrip uji agar keduanya memakai checkpoint yang sama. Variabel ini menerima folder lokal atau ID model di Hugging Face. Simpan bobot dan tokenizer mBART-50 bersama dalam hasil ekspor fine-tuning.

```powershell
$env:MBART_MODEL_ID = '.\checkpoints\mbart-python'
```

Hapus variabel dari sesi PowerShell untuk kembali ke model bawaan dengan `Remove-Item Env:MBART_MODEL_ID`. Tampilan panduan mencatat rujukan model yang digunakan.

Untuk mencoba adapter LoRA lokal, pasang dependensi eksperimen dari `requirements-training.txt` dan arahkan `MBART_MODEL_ID` ke subfolder `adapter` hasil training. Loader membaca ID serta revisi model dasar dari `adapter_config.json`. Lihat [catatan eksperimen dan serah terima model](docs/model-handoff.md) untuk contoh perintah dan hasil evaluasi.

## Aktifkan preprocessing Gemini

Buat API key di [Google AI Studio](https://aistudio.google.com/apikey). Jika `.env` belum ada, salin `.env.example` ke `.env` di akar repo, lalu isi `GEMINI_API_KEY` dengan key tersebut. Berkas `.env` diabaikan oleh Git.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m scripts.check_gemini
```

Setelah muncul `Gemini OK`, proses video YouTube dan lihat jalur yang dipakai di **Analisis pemrosesan**. Default `GEMINI_MODEL` adalah `gemini-3.5-flash-lite`. Saat pembacaan layar tersedia, Gemini dapat menerima potongan transkrip dan crop editor. Jika tidak ada frame yang bisa dikirim, Gemini dapat membantu memilih cuplikan dari teks transkrip. mBART-50 membuat ringkasan akhir. Atur `GEMINI_ENABLED=false` untuk memproses media dan teks secara lokal.

Jika Gemini mencapai batas kuota atau penggunaan, klik **Buat Panduan** tetap memproses video dengan OCR dan peringkasan lokal sesuai bukti yang tersedia. Pesan kuota tampil saat pemrosesan dan pada halaman hasil. Kesalahan konfigurasi atau koneksi dilaporkan di detail pemrosesan.

### Pembacaan kode di layar

Streamlit memakai FFmpeg untuk mengambil sampel frame sepanjang video. Tesseract membaca editor sebelum mBART merangkum. Bukti lokal dapat memperjelas kondisi `break`, cabang Python `if`/`elif`/`else`, serta panggilan fungsi yang disebut di transkrip jika pembacaannya konsisten pada dua frame. Untuk percabangan, pipeline memeriksa indentasi agar kondisi dan perintah di bawahnya berasal dari blok yang sama. Untuk YouTube, pengambilan video dilakukan setelah pemeriksaan subtitle dan persetujuan yang diperlukan.

Aplikasi bisa menyambung ucapan yang masih berlanjut pada segmen berikutnya. Langkah yang didukung kode atau penjelasan teknis tetap bisa ditampilkan dari transkrip meskipun hasil mBART belum bisa dipakai. Jika Whisper mengenali topik pemrograman tetapi belum menemukan langkah yang dapat dipakai, aplikasi mencoba satu pemeriksaan audio tambahan.

Panduan menampilkan maksimal 10 poin. Untuk tutorial dasar string Python, aplikasi menyusun poin langsung dari istilah yang jelas pada narasi atau literal yang terbaca sama pada dua frame. Pemeriksaan literal membedakan karakter escape seperti `\\n` dan `\\t` dari backslash yang ditulis ganda atau dipertahankan dalam raw string. Poin yang sudah tersusun dari bukti ini ditampilkan langsung. mBART tetap tersedia untuk bagian yang memerlukan peringkasan.

Saat transkrip belum memberi langkah yang cukup jelas, aplikasi masih memeriksa layar secara lokal. Rangkaian percabangan Python beserta aksinya dapat menjadi sumber langkah jika terbaca konsisten pada dua frame dekat. Jika mBART tidak bisa dimuat, poin yang sudah didukung sumber tetap ditampilkan.

OCR aktif otomatis jika Tesseract dan data bahasa `eng` tersedia. Pada Windows, aplikasi juga memeriksa lokasi instalasi standar. Atur `TESSERACT_EXE` jika binarinya berada di lokasi lain. Pada tutorial percabangan, OCR mencoba beberapa area untuk menemukan editor. Gunakan `VIDEO_OCR_ROI` untuk menetapkan satu area crop lewat empat koordinat relatif `x1 y1 x2 y2`, atau set `VIDEO_OCR_ENABLED=false` di lingkungan proses untuk melewati frame. [Panduan pemasangan Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)

Perbandingan transkrip, OCR lokal, dan Gemini tersedia di [notebook terpadu](notebooks/Pipeline_NLP_Video_Pemrograman.ipynb) dengan checkpoint mBART yang sama.

## Pengujian

```powershell
& $projectPython -m scripts.check_mbart
& $projectPython -m unittest discover -s tests -v
```

Skrip `check_mbart` memakai loader dan konfigurasi generasi dari aplikasi. `Tokenizer OK` dan `Model OK` mengonfirmasi pemuatan dan satu generasi di CPU, termasuk token bahasa Indonesia pada masukan dan keluaran. Untuk uji browser, proses tautan YouTube, periksa ringkasan dan sumber transkrip, lalu pilih satu segmen dan tekan **Putar segmen**.

Pada uji lokal, checkpoint dasar revisi `4ef55a20b36c6903b832e38f0604ab4bdf22c7d6` berhasil menjalankan generasi di CPU. Klip contoh pandas bernarasi sintetis menghasilkan tiga langkah di Chrome yang dapat diputar dari penanda `00:00`, `00:10`, dan `00:19`. Model dasar mengulang teks masukan pada contoh skrip uji. Perbandingan kualitas masih memerlukan data referensi dan checkpoint hasil fine-tuning.

## Pelatihan dan evaluasi

Laporan eksperimen tersedia dalam [satu notebook terpadu](notebooks/Pipeline_NLP_Video_Pemrograman.ipynb). Lihat [panduan notebook](notebooks/README.md) untuk menjalankan ulang pelatihan mBART, Whisper, dan evaluasi video.

Untuk menguji fine-tuning mBART di GPU lokal, gunakan interpreter ROCm yang sudah lolos tes GPU. Percobaan ini memakai checkpoint dasar dari cache Hugging Face dan membaca `train.csv` serta `validation.csv` dari `data/processed/indosum`.

```powershell
.\.venv-rocm\Scripts\python.exe -u -m scripts.train_mbart_pilot --output-dir outputs/mbart-pilot-ulang
```

Skrip melatih adapter LoRA pada layer terakhir decoder dengan 100 contoh latih, 20 contoh validasi, 1 epoch, dan batch size 1. Folder keluaran harus baru dan berisi adapter, `metrics.json`, serta `predictions.json`. Skor ROUGE adalah rata-rata F1 dari sampel validasi. Adapter disimpan untuk eksperimen dan memerlukan checkpoint dasar yang sama saat dimuat ulang dengan PEFT.

Gunakan batas 1.024 token masukan dan 128 token keluaran untuk percobaan konteks panjang. Opsi `--compare-base` menilai mBART tanpa adapter pada split dan batas token yang sama, lalu menyimpan hasil di `baseline_metrics.json`.

```powershell
.\.venv-rocm\Scripts\python.exe -u -m scripts.train_mbart_pilot --source-tokens 1024 --target-tokens 128 --compare-base --output-dir outputs/mbart-pilot-panjang
```

Gunakan `--train-samples` dan `--decoder-layers` untuk menguji volume data serta jumlah layer terakhir decoder yang diberi LoRA. Pertahankan sampel validasi dan batas token ketika membandingkan hasil. Setelah konfigurasi dipilih, `scripts.evaluate_mbart` dapat menjalankan pembanding model dasar dan adapter pada split `test`.

1. Pasang paket eksperimen dengan `& $projectPython -m pip install -r requirements-training.txt`. Lihat [panduan dataset](data/README.md) untuk split IndoSum dan tutorial, sampel ringkasan kode multibahasa, serta data evaluasi transkripsi.
2. Untuk evaluasi tutorial Python dan pandas, anotasi `video_id`, ringkasan referensi, istilah teknis, dan rentang waktu demonstrasi. Pisahkan data latih, validasi, dan uji berdasarkan video utuh.
3. Laporkan model yang digunakan, ROUGE-1, ROUGE-2, ROUGE-L, waktu proses, cakupan langkah, ketepatan istilah dan parameter Python, penggunaan cuplikan sumber, serta ketepatan penanda waktu.
4. Bandingkan panduan dengan transkrip biasa menggunakan tugas pencarian yang sama. Acak urutan kondisi. Ukur waktu hingga peserta menemukan demonstrasi yang benar dan laporkan persentase keberhasilan per kondisi.
