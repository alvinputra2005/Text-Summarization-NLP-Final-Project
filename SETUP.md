# Setup lokal Panduan Pemrograman

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

Untuk tab **Link YouTube**, gunakan koneksi internet dan Node.js 22 atau lebih baru di PATH. Jalankan `node --version` untuk memeriksanya. Paket `yt-dlp` dipasang bersama `requirements.txt`.

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

Jika ingin mengaktifkan Gemini, isi `GEMINI_API_KEY` di `.env` sesuai [panduan Gemini](README.md#aktifkan-preprocessing-gemini), lalu jalankan `& $projectPython -m scripts.check_gemini` untuk menguji koneksi dengan transkrip sintetis.

Untuk membaca kode yang terlihat di video, pasang Tesseract dengan data bahasa `eng`. Jalankan `tesseract --list-langs` untuk memeriksanya. Bila binari berada di lokasi khusus, atur `TESSERACT_EXE` ke path binari sebelum menjalankan Streamlit. [Panduan pemasangan Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)

```powershell
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
```

- Tempel tautan video publik bernarasi dan tekan **Buat Panduan**. Di **Hasil tiap tahap**, periksa checkpoint mBART dan sumber transkrip dari subtitle YouTube atau Whisper.
- Di **Analisis pemrosesan**, periksa apakah preprocessing memakai Gemini atau alur lokal.
- Periksa jumlah frame dan langkah yang mendapat konteks tambahan dari OCR. Jika API Gemini mencapai batas penggunaan, aplikasi menampilkan peringatan dan meneruskan pembuatan panduan secara lokal.
- Jika subtitle Indonesia belum ditemukan, **Lanjut dengan Whisper** memulai transkripsi audio. **Batal** kembali ke pilihan sumber.
- Periksa **Poin utama** di samping video. Untuk memeriksa sumber, pilih segmen dari **Hasil tiap tahap** dan tekan **Putar segmen**.
- Pilih **Ganti video** dan pastikan formulir YouTube tampil kembali dengan kolom tautan kosong.

`.gitignore` mencakup lingkungan Python, unggahan, checkpoint, keluaran, video contoh lokal, dan rahasia konfigurasi. Jika model gagal dimuat, periksa akses ke Hugging Face, ruang cache, serta kelengkapan tokenizer dan bobot pada checkpoint yang dipilih.
