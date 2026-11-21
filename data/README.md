# Dataset untuk eksperimen NLP

File unduhan ada di `data/raw/`. File siap dimuat ada di `data/processed/`. Kedua folder diabaikan Git. Jalankan perintah dari akar repo di PowerShell.

| Dataset | Hasil lokal | Kegunaan |
| --- | --- | --- |
| IndoSum | `processed/indosum/{train,validation,test}.csv` | Eksperimen mBART pada fold resmi `01`. Tersedia 14.261 contoh latih, 750 validasi, dan 3.762 uji. |
| Liputan6 | `processed/liputan6/{train,validation,test}.csv` | Pembanding mBART pada split Canonical dari paket IndoNLG. Tersedia 193.003 contoh latih, 10.960 validasi, dan 10.970 uji. |
| FLEURS Indonesia | `processed/fleurs/test.csv` dan `audio/` | Evaluasi Whisper dengan 687 audio WAV pada split uji resmi `id_id`, serta transkrip asli dan versi normalisasi FLEURS. |
| LUMINA v4 | `processed/lumina/test.csv` dan `clips/` | Uji Whisper dan unggah MP4. Sampel 48 klip dari empat penutur dengan transkrip resmi. |

IndoSum dan Liputan6 memiliki kolom `id`, `text`, dan `summary`. Skrip meratakan token yang sudah dipisahkan oleh sumber, merapikan spasi, dan menyaring pasangan kosong atau duplikat eksak dalam setiap korpus. Jika artikel atau ID muncul pada lebih dari satu split, prioritasnya `test`, lalu `validation`, lalu `train`. Jalankan eksperimen per korpus untuk mempertahankan split yang terdokumentasi.

Di manifest FLEURS, `audio_path` relatif terhadap folder CSV. Di manifest LUMINA, `video_path` juga relatif terhadap folder CSV yang bersangkutan. Klip LUMINA menyatukan video dengan WAV resmi berdasarkan ID klip. Teks referensi diambil dari Excel resmi berdasarkan nomor kalimat. Jika audio lebih panjang, frame terakhir video ditahan hingga audio selesai. Klip hasilnya bisa langsung diunggah ke Streamlit.

Keempat sumber memberi referensi untuk eksperimen berita atau pengenalan ucapan umum. Pengujian kualitas ringkasan per langkah dan akurasi penanda waktu pada tutorial pandas memerlukan referensi khusus domain tersebut. Dataset Mozilla yang dibahas sebelumnya mensyaratkan permintaan akses. FLEURS dan LUMINA dipakai untuk evaluasi ucapan bahasa Indonesia. [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu1ieso2001bo107h8377g5v)

## Muat data peringkasan

```python
from datasets import load_dataset

corpus = "indosum"  # Gunakan "liputan6" untuk percobaan lain.
splits = ("train", "validation", "test")
dataset = load_dataset("csv", data_files={
    split: f"data/processed/{corpus}/{split}.csv" for split in splits
})
```

Gunakan `text` sebagai masukan mBART dan `summary` sebagai target. Untuk Whisper, gunakan media yang dirujuk manifest dan `transcript` sebagai referensi. Normalisasikan referensi dan prediksi dengan aturan yang sama sebelum menghitung WER atau CER.

## Siapkan ulang di komputer lain

```powershell
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements-training.txt
New-Item -ItemType Directory -Force data/raw/indosum,data/raw/liputan6,data/raw/fleurs
curl.exe -fL 'https://drive.usercontent.google.com/download?id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco&export=download&confirm=t' -o data/raw/indosum/indosum.tar.gz
curl.exe -fL 'https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip' -o data/raw/liputan6/indonlg-downstream.zip
curl.exe -fL 'https://huggingface.co/datasets/google/fleurs/resolve/94d6bc3ce99786003b4c2432bee7f87bf0732e77/id_id/test-00000-of-00001.parquet' -o data/raw/fleurs/test.parquet
& $projectPython -m scripts.fetch_lumina
& $projectPython -m scripts.prepare_datasets
```

## Sumber dan ketentuan pakai

- **IndoSum.** Gunakan sitasi Kurniawan dan Louvan (2018). README di dalam arsip korpus menyatakan CC BY-SA 4.0. Repo kode mencantumkan Apache 2.0. [Repo resmi](https://github.com/kata-ai/indosum) · [Paper](https://arxiv.org/abs/1810.05334)
- **Liputan6.** Gunakan untuk riset akademik nonkomersial. Pengelola juga membatasi pemakaian komersial model yang dilatih dengan korpus dan meminta peneliti menghindari penyebaran ulang data. Berkas berasal dari distribusi IndoNLG yang dirujuk oleh loader SEACrowd. Sitasi Koto, Lau, dan Baldwin (2020). [Ketentuan sumber](https://github.com/fajri91/sum_liputan6#obtaining-liputan6-data) · [Loader SEACrowd](https://huggingface.co/datasets/SEACrowd/liputan6/blob/main/liputan6.py)
- **FLEURS.** CC BY 4.0. Gunakan atribusi pembuat dataset. [Kartu dataset Google](https://huggingface.co/datasets/google/fleurs)
- **LUMINA v4.** CC BY 4.0. Gunakan atribusi Setyaningsih, Chen, Kristian, Handayani, dan Irianto (2024) serta sebutkan bahwa WAV dipasang pada video untuk paket pengujian lokal. [Mendeley Data](https://data.mendeley.com/datasets/8fw93k4rny/4)
