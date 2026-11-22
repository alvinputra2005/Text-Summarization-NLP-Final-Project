# Panduan Pemrograman

Aplikasi Streamlit lokal yang meringkas video tutorial pemrograman berbahasa Indonesia. Pilih **Unggah MP4** atau tempel alamat video publik di **Link YouTube**, lalu tekan **Buat Panduan**. Pengujian video berfokus pada Python.

Pengantar dan poin utama muncul di samping video. **Analisis pemrosesan** menampilkan alur, keluaran, dan waktu proses. Di **Skor acuan**, isi ringkasan acuan untuk membandingkan ROUGE-1, ROUGE-2, ROUGE-3, dan ROUGE-L, atau transkrip acuan untuk menghitung WER dan CER. Isi ringkasan pembanding jika ingin memakai keluaran model lain. Di **Audit langkah**, putar segmen sumber, periksa istilah teknis, dan catat penilaian selama sesi. Bila narasi kosong atau materinya belum bisa dikenali, pemutar tetap tersedia bersama status pemeriksaan.

## Pipeline

1. Untuk MP4, FFprobe memeriksa video dan jalur audio. FFmpeg menyiapkan media untuk pemutar dan dekode audio.
2. Untuk YouTube, aplikasi memeriksa subtitle Indonesia dan meminta keputusan sebelum mengambil audio jika subtitle belum ditemukan. Jika subtitle panjang sulit dibaca, aplikasi mencoba transkripsi ulang dan memilih hasil yang lebih terbaca. MP4 memakai Whisper langsung.
3. Aplikasi memilih dan merapikan cuplikan narasi. Jika Gemini aktif, Gemini memilih bagian yang berisi inti tutorial dan aplikasi memeriksa hasilnya terhadap transkrip asli. Alur lokal mengelompokkan narasi berdasarkan jeda dan perubahan tindakan serta mengenali sejumlah pola demonstrasi Python.
4. mBART-50 meringkas teks tiap langkah. Aplikasi memeriksa detail penting dan keterbacaan keluarannya. Bagian yang sulit diringkas tetap bisa diputar dan dilihat di transkrip. Waktu mulai dan selesai berasal dari segmen transkrip.
5. Streamlit menampilkan panduan, transkrip bertanda waktu, dan durasi proses. Skor WER, CER, dan ROUGE memakai teks acuan yang diberikan pengguna.

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

Jika Python 3.11 dipasang melalui `uv`, gunakan `uv venv --python 3.11 --seed .venv` untuk membuat lingkungan. Buka alamat lokal yang ditampilkan Streamlit. Untuk uji awal di CPU, pilih MP4 pendek. Pengunduhan pertama memerlukan koneksi internet dan ruang cache. Bobot PyTorch model dasar berukuran sekitar 2,44 GB. [Berkas checkpoint mBART](https://huggingface.co/facebook/mbart-large-50/tree/main)

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

Setelah muncul `Gemini OK`, proses MP4 atau video YouTube kembali dan lihat nama model di **Analisis pemrosesan → Preprocessing**. Default `GEMINI_MODEL` adalah `gemini-3.1-flash-lite`. Saat Gemini aktif, potongan teks transkrip dikirim ke Google untuk memilih inti tutorial dan merapikan bahasa lisan. mBART-50 membuat ringkasan akhir dari cuplikan yang lolos pemeriksaan. Atur `GEMINI_ENABLED=false` untuk menjalankan semuanya secara lokal. Aplikasi juga beralih ke preprocessing lokal jika API gagal atau seluruh cuplikan gagal diverifikasi.

## Pengujian

```powershell
& $projectPython -m scripts.check_mbart
& $projectPython -m unittest discover -s tests -v
```

Skrip `check_mbart` memakai loader dan konfigurasi generasi dari aplikasi. `Tokenizer OK` dan `Model OK` mengonfirmasi pemuatan dan satu generasi di CPU, termasuk token bahasa Indonesia pada masukan dan keluaran. Untuk uji browser, proses MP4 pendek atau link YouTube, periksa ringkasan dan sumber transkrip, lalu pilih satu segmen dan tekan **Putar segmen**.

Pada uji lokal, checkpoint dasar revisi `4ef55a20b36c6903b832e38f0604ab4bdf22c7d6` berhasil menjalankan generasi di CPU. Klip contoh pandas bernarasi sintetis menghasilkan tiga langkah di Chrome yang dapat diputar dari penanda `00:00`, `00:10`, dan `00:19`. Model dasar mengulang teks masukan pada contoh skrip uji. Perbandingan kualitas masih memerlukan data referensi dan checkpoint hasil fine-tuning.

## Pelatihan dan evaluasi

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
