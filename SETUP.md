# Setup lokal Panduan Praktik Python

[README.md](README.md) memuat pipeline, instalasi dependensi, konfigurasi `MBART_MODEL_ID`, dan cara menjalankan Streamlit. Ikuti panduan ini untuk memeriksa setiap bagian di Windows PowerShell.

## 1. Periksa prasyarat

Gunakan Python 3.11 dan Git. Pasang FFmpeg jika `ffmpeg` atau `ffprobe` belum tersedia, lalu buka ulang PowerShell.

```powershell
winget install --exact --id Gyan.FFmpeg
```

```powershell
py -3.11 --version
git --version
ffmpeg -version
ffprobe -version
```

## 2. Uji dependensi dan checkpoint

Setelah membuat `.venv` dan memasang dependensi sesuai README, jalankan dari akar repo.

```powershell
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip check
& $projectPython -m scripts.check_mbart
& $projectPython -m unittest discover -s tests -v
```

Skrip memakai `facebook/mbart-large-50` jika `MBART_MODEL_ID` kosong. Untuk checkpoint hasil fine-tuning, arahkan variabel itu ke folder yang berisi bobot dan tokenizer mBART-50, atau ke ID modelnya di Hugging Face. Periksa bahwa keluaran mencantumkan checkpoint yang diharapkan, `Tokenizer OK` dengan `id_ID`, dan `Model OK`.

Untuk adapter LoRA lokal, pasang `requirements-training.txt` dan gunakan subfolder `adapter` sebagai `MBART_MODEL_ID`. Cara menguji dan membandingkan adapter tersedia di [catatan serah terima model](docs/model-handoff.md).

## 3. Uji FFprobe dan Whisper

Simpan MP4 bernarasi pendek di `samples/local/tutorial.mp4`. Pastikan FFprobe menampilkan jalur `video` dan `audio`, lalu periksa bahwa Whisper menghasilkan teks dengan `start < end`.

```powershell
ffprobe -v error -show_entries stream=index,codec_type,codec_name -of json '.\samples\local\tutorial.mp4'
& $projectPython -m whisper '.\samples\local\tutorial.mp4' --model base --language Indonesian --task transcribe --device cpu --fp16 False --output_format json --output_dir outputs
$transcript = Get-Content '.\outputs\tutorial.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$transcript.segments | Select-Object -First 3 start, end, text | Format-Table -Wrap
```

## 4. Uji aplikasi

```powershell
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
```

- Unggah MP4 dan tekan **Buat Panduan**. Periksa bahwa model yang ditampilkan sesuai konfigurasi.
- Klik setiap waktu dan klik ulang salah satu penanda. Pastikan pemutar berpindah ke waktu sumber tanpa tahap transkripsi atau peringkasan berjalan lagi.
- Buka **Lihat transkrip** dan cocokkan ucapan sumber dengan langkahnya.
- Pilih **Ganti video** dan pastikan hasil sebelumnya hilang. Uji pesan untuk MP4 rusak atau tanpa audio.

`.gitignore` mencakup lingkungan Python, unggahan, checkpoint, keluaran, video contoh lokal, dan rahasia konfigurasi. Jika model gagal dimuat, periksa akses ke Hugging Face, ruang cache, serta kelengkapan tokenizer dan bobot pada checkpoint yang dipilih.
