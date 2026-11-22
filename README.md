# Panduan Praktik Python

Aplikasi Streamlit lokal yang membuat panduan bertanda waktu dari video tutorial Python berbahasa Indonesia. Pilih **Unggah MP4** atau tempel alamat video publik di **Link YouTube**, lalu tekan **Buat Panduan**.

Pengantar dan langkah muncul di samping video. Klik penanda waktu untuk melompat ke bagian yang menjelaskan tindakan dan hasilnya. **Analisis pemrosesan** menampilkan alur, keluaran, dan waktu proses. Di **Skor acuan**, isi ringkasan acuan untuk membandingkan ROUGE-1, ROUGE-2, ROUGE-3, dan ROUGE-L, atau transkrip acuan untuk menghitung WER dan CER. ROUGE memakai gabungan teks langkah dalam urutan video. Isi ringkasan pembanding jika ingin memakai keluaran model lain. Di **Audit langkah**, periksa istilah teknis, segmen sumber, dan catat penilaian selama sesi.

## Pipeline

1. Untuk MP4, FFprobe memeriksa video dan jalur audio. FFmpeg menyiapkan media untuk pemutar dan dekode audio.
2. Untuk YouTube, aplikasi mencoba mengambil subtitle Indonesia dari video secara otomatis. Jika gagal atau subtitle tidak tersedia, `yt-dlp` mengambil audio agar Whisper membuat transkrip. MP4 memakai Whisper langsung.
3. Aplikasi mengelompokkan narasi berdasarkan jeda dan perubahan tindakan. Pola khusus mengenali demonstrasi `input()` dan konversi tipe dari ucapan pemateri, termasuk hasil percobaan yang terdeteksi.
4. mBART-50 meringkas teks tiap langkah. Jika hasilnya kehilangan detail penting, aplikasi memakai teks yang disiapkan dari segmen sumber. Waktu mulai dan selesai berasal dari segmen transkrip.
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

## Konfigurasi model

Default aplikasi adalah `facebook/mbart-large-50`. `MBart50TokenizerFast` menggunakan `id_ID` sebagai bahasa sumber dan target. Generasi dimulai dengan token pembuka dekoder `eos`, lalu memaksa token bahasa target `id_ID`. Checkpoint dasar merupakan model pralatih untuk uji pemuatan dan generasi. Gunakan ringkasan referensi tutorial untuk menilai kualitas peringkasan model yang sudah di-fine-tune. [Kartu model](https://huggingface.co/facebook/mbart-large-50) · [Dokumentasi Transformers](https://huggingface.co/docs/transformers/model_doc/mbart)

Atur `MBART_MODEL_ID` sebelum menjalankan aplikasi atau skrip uji agar keduanya memakai checkpoint yang sama. Variabel ini menerima folder lokal atau ID model di Hugging Face. Simpan bobot dan tokenizer mBART-50 bersama dalam hasil ekspor fine-tuning.

```powershell
$env:MBART_MODEL_ID = '.\checkpoints\mbart-python'
```

Hapus variabel dari sesi PowerShell untuk kembali ke model bawaan dengan `Remove-Item Env:MBART_MODEL_ID`. Tampilan panduan mencatat rujukan model yang digunakan.

Untuk mencoba adapter LoRA lokal, pasang dependensi eksperimen dari `requirements-training.txt` dan arahkan `MBART_MODEL_ID` ke subfolder `adapter` hasil training. Loader membaca ID serta revisi model dasar dari `adapter_config.json`. Lihat [catatan eksperimen dan serah terima model](docs/model-handoff.md) untuk contoh perintah dan hasil evaluasi.

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

1. Pasang paket eksperimen dengan `& $projectPython -m pip install -r requirements-training.txt`. Lihat [panduan dataset](data/README.md) untuk split IndoSum dan Liputan6 yang siap dimuat, serta audio FLEURS dan video LUMINA dengan transkrip referensi.
2. Untuk evaluasi tutorial Python dan pandas, anotasi `video_id`, ringkasan referensi, istilah teknis, dan rentang waktu demonstrasi. Pisahkan data latih, validasi, dan uji berdasarkan video utuh.
3. Laporkan model yang digunakan, ROUGE-1, ROUGE-2, ROUGE-L, waktu proses, cakupan langkah, ketepatan istilah dan parameter Python, penggunaan cuplikan sumber, serta ketepatan penanda waktu.
4. Bandingkan panduan dengan transkrip biasa menggunakan tugas pencarian yang sama. Acak urutan kondisi. Ukur waktu hingga peserta menemukan demonstrasi yang benar dan laporkan persentase keberhasilan per kondisi.
