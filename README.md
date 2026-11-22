# Panduan Praktik Python

Aplikasi Streamlit lokal yang mengubah video tutorial Python berbahasa Indonesia menjadi panduan praktik dengan penanda waktu. Pilih **Unggah MP4** untuk memilih berkas video, atau pilih **Link YouTube** dan tempel alamat satu video publik bernarasi. Tekan **Buat Panduan** agar Whisper mentranskripsikan audio dan mBART meringkasnya. Lalu cari kata seperti `dropna` di **Cari langkah** dan tekan Enter.

Klik penanda waktu untuk memutar langkah, buka **Lihat transkrip** untuk memeriksa teks sumber, atau buka **Detail pemrosesan** untuk melihat sumber transkrip, model, dan durasi proses. Untuk video yang membatasi pemutaran sematan, **Buka di YouTube** membuka video pada waktu langkah yang dipilih. Gunakan **Ganti video** untuk memproses sumber lain dalam sesi browser.

## Pipeline

1. Untuk MP4, FFprobe memeriksa video dan jalur audio. FFmpeg menyiapkan media untuk pemutar dan dekode audio.
2. Untuk YouTube, `yt-dlp` mengambil audio ke folder sementara. Whisper `base` multibahasa menghasilkan transkrip dan penanda waktu dari kedua sumber.
3. Python mengelompokkan potongan yang membahas satu tindakan dan menyimpan rentang waktu serta transkrip sumber.
4. mBART-50 merangkum tiap kelompok teks. Jika hasil kosong atau pemeriksaan mendeteksi detail penting yang berubah, aplikasi menampilkan **Cuplikan sumber**.
5. Streamlit menautkan ringkasan ke waktu mulai dari Whisper. Klik penanda menggeser pemutar menggunakan hasil sesi yang sudah tersedia.

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

Skrip `check_mbart` memakai loader dan konfigurasi generasi dari aplikasi. `Tokenizer OK` dan `Model OK` mengonfirmasi pemuatan dan satu generasi di CPU, termasuk token bahasa Indonesia pada masukan dan keluaran. Untuk uji browser, unggah MP4 pendek, periksa bahwa setiap tombol waktu membawa video ke langkah sumber, lalu coba klik ulang, buka transkrip, dan ganti unggahan.

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
