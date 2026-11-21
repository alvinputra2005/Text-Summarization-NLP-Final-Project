# Panduan Praktik Python

Aplikasi Streamlit lokal yang mengubah video tutorial Python berbahasa Indonesia menjadi panduan praktik dengan penanda waktu. Unggah MP4 bernarasi, tekan **Buat Panduan**, lalu klik waktu pada langkah yang ingin diputar. Buka **Lihat transkrip** untuk memeriksa ucapan sumber. Gunakan **Ganti video** untuk memproses unggahan lain dalam sesi browser.

## Pipeline

1. FFprobe memeriksa video dan jalur audio. FFmpeg menyiapkan media untuk pemutar dan dekode audio.
2. Whisper `base` multibahasa mentranskripsi narasi Indonesia dan menghasilkan `start` serta `end` dalam detik.
3. Python mengelompokkan potongan yang membahas satu tindakan dan menyimpan rentang waktu serta transkrip sumber.
4. mBART-50 merangkum tiap kelompok teks. Jika hasil kosong atau pemeriksaan mendeteksi detail penting yang berubah, aplikasi menampilkan **Cuplikan sumber**.
5. Streamlit menautkan ringkasan ke waktu mulai dari Whisper. Klik penanda menggeser pemutar menggunakan hasil sesi yang sudah tersedia.

## Jalankan di Windows PowerShell

Pasang Python 3.11, FFmpeg, dan FFprobe. Lihat [SETUP.md](SETUP.md) untuk prasyarat dan cara menguji audio. Jalankan dari akar repo.

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

## Konfigurasi model

Default aplikasi adalah `facebook/mbart-large-50`. `MBart50TokenizerFast` menggunakan `id_ID` sebagai bahasa sumber dan target. Generasi dimulai dengan token pembuka dekoder `eos`, lalu memaksa token bahasa target `id_ID`. Checkpoint dasar merupakan model pralatih untuk uji pemuatan dan generasi. Gunakan ringkasan referensi tutorial untuk menilai kualitas peringkasan model yang sudah di-fine-tune. [Kartu model](https://huggingface.co/facebook/mbart-large-50) · [Dokumentasi Transformers](https://huggingface.co/docs/transformers/model_doc/mbart)

Atur `MBART_MODEL_ID` sebelum menjalankan aplikasi atau skrip uji agar keduanya memakai checkpoint yang sama. Variabel ini menerima folder lokal atau ID model di Hugging Face. Simpan bobot dan tokenizer mBART-50 bersama dalam hasil ekspor fine-tuning.

```powershell
$env:MBART_MODEL_ID = '.\checkpoints\mbart-python'
```

Hapus variabel dari sesi PowerShell untuk kembali ke model bawaan dengan `Remove-Item Env:MBART_MODEL_ID`. Tampilan panduan mencatat rujukan model yang digunakan.

## Pengujian

```powershell
& $projectPython -m scripts.check_mbart
& $projectPython -m unittest discover -s tests -v
```

Skrip `check_mbart` memakai loader dan konfigurasi generasi dari aplikasi. `Tokenizer OK` dan `Model OK` mengonfirmasi pemuatan dan satu generasi di CPU, termasuk token bahasa Indonesia pada masukan dan keluaran. Untuk uji browser, unggah MP4 pendek, periksa bahwa setiap tombol waktu membawa video ke langkah sumber, lalu coba klik ulang, buka transkrip, dan ganti unggahan.

Pada uji lokal, checkpoint dasar revisi `4ef55a20b36c6903b832e38f0604ab4bdf22c7d6` berhasil menjalankan generasi di CPU. Klip contoh pandas bernarasi sintetis menghasilkan tiga langkah di Chrome yang dapat diputar dari penanda `00:00`, `00:10`, dan `00:19`. Model dasar mengulang teks masukan pada contoh skrip uji. Perbandingan kualitas masih memerlukan data referensi dan checkpoint hasil fine-tuning.

## Pelatihan dan evaluasi

1. Pasang paket eksperimen dengan `& $projectPython -m pip install -r requirements-training.txt`. Gunakan [IndoSum](https://github.com/kata-ai/indosum) untuk eksperimen awal peringkasan bahasa Indonesia.
2. Untuk evaluasi tutorial Python dan pandas, anotasi `video_id`, ringkasan referensi, istilah teknis, dan rentang waktu demonstrasi. Pisahkan data latih, validasi, dan uji berdasarkan video utuh.
3. Laporkan model yang digunakan, ROUGE-1, ROUGE-2, ROUGE-L, waktu proses, cakupan langkah, ketepatan istilah dan parameter Python, penggunaan cuplikan sumber, serta ketepatan penanda waktu.
4. Bandingkan panduan dengan transkrip biasa menggunakan tugas pencarian yang sama. Acak urutan kondisi. Ukur waktu hingga peserta menemukan demonstrasi yang benar dan laporkan persentase keberhasilan per kondisi.
