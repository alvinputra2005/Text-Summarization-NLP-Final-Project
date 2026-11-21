# Panduan setup Python Tutorial Summarizer

Untuk menjalankan aplikasi yang sudah tersedia di folder ini, ikuti [petunjuk README](README.md#jalankan-di-windows-powershell). Bagian di bawah mendokumentasikan setup awal dan cara menguji tiap komponen secara terpisah.

Panduan ini menyiapkan proyek di **Windows PowerShell**. Aplikasi yang akan dikembangkan menerima video tutorial Python berbahasa Indonesia, mengubah audio menjadi transkrip bertanda waktu, lalu merangkum instruksi per langkah. Pengguna dapat mengeklik waktu pada panduan untuk membuka bagian video yang sesuai.

**Target setup pertama** adalah repo GitHub yang bisa diklon, halaman Streamlit yang bisa dibuka, satu MP4 yang berhasil ditranskripsi, dan IndoBART-v2 yang berhasil dimuat serta diuji dengan satu masukan. Halaman Streamlit di panduan ini hanya memeriksa unggahan. Integrasi seluruh pipeline dilakukan setelah tiap komponen lolos uji.

## 1. Teknologi dan perannya

| Bagian | Teknologi | Peran |
|---|---|---|
| Bahasa dan antarmuka | Python 3.11, Streamlit | Unggah MP4, tampilkan panduan praktik, dan putar video dari waktu yang dipilih |
| Validasi media | FFprobe | Periksa apakah berkas terbaca serta memiliki jalur video dan audio |
| Dekode audio | FFmpeg | Baca audio dari MP4 untuk Whisper |
| Transkripsi | Whisper, PyTorch | Hasilkan teks beserta waktu mulai dan selesai per segmen |
| Segmentasi | Python | Bersihkan transkrip dan kelompokkan potongan yang membahas satu langkah |
| Tokenisasi | IndoNLGTokenizer, SentencePiece | Siapkan ID token dan attention mask untuk IndoBART-v2 |
| Ringkasan | IndoBART-v2, Transformers, PyTorch | Ringkas setiap segmen menjadi instruksi singkat |
| Pelatihan dan evaluasi | Datasets, Accelerate, JiWER, ROUGE | Siapkan data, lakukan fine-tuning, dan ukur kualitas hasil |

**Pembagian kerja pipeline:** Whisper menentukan waktu dari audio. IndoBART-v2 bekerja pada teks segmen. Python memasangkan ringkasan dengan waktu dari segmen sumber. Saat pengguna memilih langkah, Streamlit cukup membaca waktu yang sudah disimpan.

## 2. Pasang prasyarat

1. Pasang [Python 3.11](https://www.python.org/downloads/) beserta Python Launcher untuk Windows.
2. Pasang [Git for Windows](https://git-scm.com/install/windows). Siapkan akun GitHub dengan akses tulis ke repo tim.
3. Pasang FFmpeg melalui PowerShell menggunakan Windows Package Manager:

   ```powershell
   winget install --exact --id Gyan.FFmpeg
   ```

   Kalau `winget` tidak tersedia, gunakan build Windows dari [halaman unduhan FFmpeg](https://ffmpeg.org/download.html), ekstrak arsipnya, lalu tambahkan folder `bin` yang berisi `ffmpeg.exe` dan `ffprobe.exe` ke **Path** pada **Environment Variables** untuk akunmu.

4. Tutup dan buka kembali PowerShell, lalu jalankan:

   ```powershell
   py -3.11 --version
   git --version
   ffmpeg -version
   ffprobe -version
   ```

Gunakan CPU untuk uji awal. Koneksi internet diperlukan saat mengunduh paket dan bobot model untuk pertama kali. Unduhan model bisa memakan beberapa menit, tergantung koneksi.

## 3. Buat atau klon repo

**Jika repo tim sudah tersedia**, klon repo tersebut. Ganti `USERNAME-ATAU-ORG` dengan pemilik repo.

```powershell
git clone https://github.com/USERNAME-ATAU-ORG/python-tutorial-summarizer.git
Set-Location python-tutorial-summarizer
```

**Jika kamu mulai dari folder lokal baru**, buat repo kosong di GitHub dengan nama `python-tutorial-summarizer`. Biarkan opsi pembuatan README, lisensi, dan `.gitignore` kosong jika ketiga file itu akan dibuat dari folder lokal. Lalu buat folder dan repo lokal:

```powershell
New-Item -ItemType Directory -Path python-tutorial-summarizer
Set-Location python-tutorial-summarizer
git init -b main
```

Setelah itu, letakkan dokumen ini sebagai `SETUP.md` di akar repo. Perintah berikutnya dijalankan dari folder yang sama.

## 4. Siapkan file proyek

Struktur target setelah pipeline mulai diintegrasikan:

```text
python-tutorial-summarizer/
├── app.py
├── src/
│   ├── media.py
│   ├── transcription.py
│   ├── segmentation.py
│   └── summarization.py
├── scripts/
│   └── check_indobart.py
├── notebooks/
├── requirements.txt
├── requirements-training.txt
├── .gitignore
├── README.md
└── SETUP.md
```

Untuk setup, cukup buat file yang isinya disediakan di bawah. Modul `src/` dan notebook dapat ditambahkan saat bagiannya mulai dikerjakan. Isi awal `README.md` cukup menjelaskan tujuan proyek dan menautkan `[Panduan setup](SETUP.md)`.

### `requirements.txt`

```text
streamlit
openai-whisper
torch>=2.6
transformers
indobenchmark-toolkit
sentencepiece
```

Daftar ini menjadi titik awal sebelum tim mengunci versi yang lolos uji. Gunakan paket PyTorch yang sesuai dengan perangkat dari [panduan instalasi resmi PyTorch](https://pytorch.org/get-started/locally/) jika diperlukan. Batas minimum PyTorch mengakomodasi pemuatan bobot format PyTorch pada Transformers yang mensyaratkan versi 2.6 atau lebih baru.

### `requirements-training.txt`

```text
-r requirements.txt
datasets
accelerate
jiwer
rouge-score
```

Paket ini diperlukan ketika mengerjakan pelatihan atau evaluasi. Jalur aplikasi dan uji model awal cukup memakai `requirements.txt`.

### `.gitignore`

```gitignore
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
.env.*
!.env.example
.streamlit/secrets.toml
uploads/
checkpoints/
outputs/
samples/local/
```

Simpan MP4 pribadi di `samples/local/`. Bobot model hasil pelatihan disimpan di `checkpoints/` atau repositori model yang aksesnya sesuai. Catat ID model atau lokasi checkpoint di README agar anggota tim memakai bobot yang sama.

## 5. Buat lingkungan Python

```powershell
py -3.11 -m venv .venv
$projectPython = ".\.venv\Scripts\python.exe"

& $projectPython --version
& $projectPython -m pip install --upgrade pip
& $projectPython -m pip install -r requirements.txt
& $projectPython -m pip check
```

Variabel `$projectPython` berlaku pada jendela PowerShell tersebut. Di terminal baru, tetapkan kembali variabelnya dari akar repo. Memanggil interpreter melalui path ini juga membantu di Windows jika aktivasi `Activate.ps1` dibatasi oleh pengaturan PowerShell.

Kalau perlu memasang alat pelatihan dan evaluasi:

```powershell
& $projectPython -m pip install -r requirements-training.txt
& $projectPython -m pip check
```

Cek impor tanpa mengunduh bobot:

```powershell
& $projectPython -c "import streamlit, whisper, torch, transformers, sentencepiece; from indobenchmark import IndoNLGTokenizer; print('Import OK'); print('PyTorch:', torch.__version__); print('CUDA aktif:', torch.cuda.is_available())"
```

`CUDA aktif: False` normal jika kamu memilih instalasi CPU.

## 6. Uji Streamlit dan unggah MP4

Buat `app.py` dengan contoh minimum ini:

```python
import streamlit as st

st.set_page_config(page_title="Python Tutorial Summarizer", layout="wide")
st.title("Ringkas video tutorial Python")
st.write("Unggah satu MP4 untuk mengecek antarmuka proyek")

video = st.file_uploader("Video tutorial", type=["mp4"])

if video is not None:
    st.write(video.name)
    st.video(video, format="video/mp4")
```

Jalankan dari akar repo:

```powershell
$projectPython = ".\.venv\Scripts\python.exe"
& $projectPython -m streamlit run app.py
```

Buka alamat **Local URL** yang tercetak di terminal. Unggah MP4 pendek dan pastikan video bisa diputar. Tekan **Ctrl+C** di terminal untuk menghentikan server. Pembatas jenis berkas pada antarmuka membantu memilih file. Jalur audio dan keterbacaan berkas diperiksa dengan FFprobe pada uji berikutnya.

## 7. Uji FFprobe dan Whisper

Pilih MP4 pendek yang berisi narasi bahasa Indonesia. Simpan sebagai `samples/local/tutorial.mp4`.

```powershell
ffprobe -v error -show_entries stream=index,codec_type,codec_name -of json ".\samples\local\tutorial.mp4"
```

Cari setidaknya satu stream dengan `codec_type` bernilai `video` dan satu lagi bernilai `audio`. Lalu jalankan Whisper:

```powershell
& $projectPython -m whisper ".\samples\local\tutorial.mp4" --model base --language Indonesian --task transcribe --device cpu --fp16 False --output_format json --output_dir outputs
```

Periksa hasilnya:

```powershell
$transcript = Get-Content ".\outputs\tutorial.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$transcript.segments | Select-Object -First 3 start, end, text | Format-Table -Wrap
```

**Lulus** jika ada segmen yang berisi teks sesuai audio serta `start < end`. Waktu dinyatakan dalam detik. Pertahankan nilai waktu itu saat menyimpan dan mengelompokkan segmen. Untuk antarmuka, format tampilan seperti `04:10` bisa dibuat dari nilai detik tersebut.

## 8. Uji tokenizer dan IndoBART-v2

Buat folder `scripts/`, lalu salin kode berikut ke `scripts/check_indobart.py`. Uji ini memeriksa pemuatan tokenizer, bobot model, dan satu *forward pass* pada CPU.

```python
import torch
from indobenchmark import IndoNLGTokenizer
from transformers import AutoModelForSeq2SeqLM

model_id = "indobenchmark/indobart-v2"
tokenizer = IndoNLGTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()

language = "[indonesian]"
inputs = tokenizer.prepare_input_for_generation(
    "Baca berkas CSV dengan pandas, lalu hapus baris yang kosong.",
    lang_token=language,
    decoder_lang_token=language,
    return_tensors="pt",
)

# Toolkit IndoNLG dapat menghasilkan tensor satu dimensi untuk satu teks.
input_ids = inputs["input_ids"].reshape(1, -1)
attention_mask = inputs["attention_mask"].reshape(1, -1)
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

with torch.inference_mode():
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

print(f"Tokenizer OK: {input_ids.shape[1]} token")
print(f"Model OK: logits {tuple(output.logits.shape)}")
```

Jalankan:

```powershell
& $projectPython ".\scripts\check_indobart.py"
```

**Lulus** jika muncul `Tokenizer OK` dan `Model OK`. Model publik ini menjadi bobot awal. Kualitas ringkasan tutorial diuji terpisah setelah integrasi dan, bila sudah tersedia, dengan checkpoint model yang telah di-*fine-tune*.

## 9. Commit dan push pertama

Kalau repo sudah diklon dan tim memakai pull request, buat branch sebelum commit dengan `git switch -c setup/project-bootstrap`. Untuk repo lokal baru, tetap gunakan `main`. Setelah memilih branch, periksa file yang akan dikirim:

```powershell
git status --short
git add SETUP.md README.md .gitignore requirements.txt requirements-training.txt app.py scripts/check_indobart.py
git diff --cached --stat
git commit -m "Set up tutorial summarizer project"
```

Kalau Git meminta identitas untuk repo ini, isi nama dan email yang ingin dipakai, lalu ulangi commit:

```powershell
git config user.name "Nama Kamu"
git config user.email "email-kamu@example.com"
```

- **Jika mulai dari repo lokal baru**, tambahkan URL remote milik tim dengan `git remote add origin https://github.com/USERNAME-ATAU-ORG/python-tutorial-summarizer.git`, cek dengan `git remote -v`, lalu jalankan `git push -u origin main`.
- **Jika repo sudah diklon**, cek remote dengan `git remote -v`. Clone sudah memasang `origin`. Jika kamu memakai branch setup, jalankan `git push -u origin setup/project-bootstrap`, lalu buka pull request di GitHub. Jika tim mengizinkan push langsung pada branch yang sedang dipakai, jalankan `git push`.

Setelah setup berhasil di satu mesin, catat versi untuk reproduksibilitas. Misalnya, jalankan `& $projectPython -m pip freeze` dari lingkungan yang hanya berisi paket aplikasi, lalu simpan hasilnya bersama dokumentasi proyek setelah tim memeriksa isinya. Catat juga OS, versi Python, FFmpeg, dan profil CPU atau CUDA.

## 10. Kalau ada kendala

| Gejala | Yang diperiksa |
|---|---|
| `py` tidak dikenali | Cek apakah Python Launcher terpasang. Buka terminal baru atau gunakan path penuh menuju Python 3.11 untuk membuat `venv`. |
| `ffmpeg` atau `ffprobe` tidak dikenali | Jalankan `Get-Command ffmpeg, ffprobe`. Pastikan folder `bin` masuk ke Path, lalu buka terminal baru. |
| Video tampil tetapi tidak memiliki audio | Jalankan FFprobe dan periksa `codec_type`. Ulangi uji dengan video bernarasi yang memiliki jalur audio. |
| Modul Python tidak ditemukan | Pastikan semua perintah pip dan Streamlit memakai `$projectPython`. Jalankan ulang `pip check` melalui interpreter yang sama. |
| `IndoNLGTokenizer` tidak dikenal | Gunakan impor eksplisit dari paket `indobenchmark-toolkit` seperti pada skrip uji. Periksa apakah paket terpasang di `.venv`. |
| Unduhan bobot gagal | Periksa akses ke Hugging Face dan ruang disk. Sesudah koneksi normal, ulangi skrip uji. |
| CPU lambat | Gunakan video pendek dan model Whisper `base` saat memeriksa alur. Catat pengujian GPU secara terpisah jika tersedia. |
| Push ditolak | Cek pemilik remote, akses tulis, dan aturan branch. Jika branch utama dilindungi, gunakan branch kerja dan pull request. |
| Deploy Streamlit gagal menemukan FFmpeg | Untuk Community Cloud, tambahkan `packages.txt` berisi satu baris `ffmpeg`, dan pilih versi Python yang sesuai dengan setup lokal. |

## 11. Kriteria setup selesai

- [ ] Python 3.11, Git, FFmpeg, dan FFprobe dikenali dari PowerShell
- [ ] Dependensi terpasang di `.venv` dan `pip check` berhasil
- [ ] Streamlit terbuka dan bisa memutar satu MP4 uji
- [ ] Whisper menghasilkan JSON berisi teks, `start`, dan `end`
- [ ] Skrip IndoBART-v2 menampilkan `Tokenizer OK` dan `Model OK`
- [ ] Berkas setup dan kode uji masuk repo GitHub
- [ ] MP4 pribadi, bobot lokal, dan rahasia konfigurasi tetap di luar commit

Setelah semua cek selesai, sambungkan pemrosesan video ke antarmuka. Buat satu fungsi untuk mengubah MP4 menjadi daftar objek berisi `start`, `end`, `transcript`, dan `summary`. Simpan daftar itu untuk sesi pengguna, lalu gunakan `start` saat pengguna memilih langkah.

## Referensi teknis

- [Setup OpenAI Whisper](https://github.com/openai/whisper)
- [Unduhan FFmpeg](https://ffmpeg.org/download.html)
- [Instalasi PyTorch](https://pytorch.org/get-started/locally/)
- [IndoBART-v2](https://huggingface.co/indobenchmark/indobart-v2)
- [Indobenchmark Toolkit dan contoh tokenizer](https://github.com/indobenchmark/indobenchmark-toolkit)
- [Dependensi Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Menambahkan kode lokal ke GitHub di Windows](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)
