# Dataset untuk eksperimen NLP

File unduhan ada di `data/raw/`. File siap dimuat ada di `data/processed/`. Kedua folder diabaikan Git. Jalankan perintah dari akar repo di PowerShell.

| Dataset | Hasil lokal | Kegunaan |
| --- | --- | --- |
| IndoSum | `processed/indosum/{train,validation,test}.csv` | Eksperimen mBART pada fold resmi `01`. Tersedia 14.261 contoh latih, 750 validasi, dan 3.762 uji. |
| FLEURS Indonesia | `processed/fleurs/test.csv` dan `audio/` | Evaluasi Whisper dengan 687 audio WAV pada split uji resmi `id_id`, serta transkrip asli dan versi normalisasi FLEURS. |
| LUMINA v4 | `processed/lumina/test.csv` dan `clips/` | Uji Whisper dan unggah MP4. Sampel 48 klip dari empat penutur dengan transkrip resmi. |
| Pembagian LUMINA untuk notebook | `processed/lumina_split/{train,validation,test}.csv` | 24/12/12 klip dari sampel LUMINA yang sama, dibagi berdasarkan penutur. |
| NSS-ID · Imperative | `processed/nss-id/{train,val,test}.csv` dan `audio/` | Sampel penutur manusia dari split resmi untuk fine-tuning dan evaluasi Whisper. |
| Audio coding untuk Whisper | `processed/whisper_video/{train,validation,test}.csv` dan `audio/` | Empat video tutorial Python dengan subtitle otomatis. Split per video menyediakan 20/12/12 klip. |
| Coding pilot | `processed/coding_pilot/{train,validation,test}.csv` | 108 contoh sintetis orisinal untuk LoRA mBART. Skenario dipisah utuh menjadi 56 latih, 24 validasi, dan 28 uji. |
| Coding video mix | `processed/coding_video_mix/{train,validation,test}.csv` | 33 cuplikan Python Indonesia beranotasi dari 12 video, 28 contoh sintetis, dan 10 cuplikan CrossTask untuk data latih tambahan. |
| The Vault Function | `processed/multilingual_code/{train,validation,test}.csv` | 50 fungsi latih dan masing-masing 20 pada validasi serta uji. Bahasa mencakup C, C#, C++, Go, Java, JavaScript, PHP, Python, Ruby, dan Rust. |
| Campuran tutorial multibahasa | `processed/coding_multilingual_mix/{train,validation,test}.csv` | Gabungan 56 contoh Coding Video Mix dan 50 contoh latih The Vault. Validasi dan uji tetap memakai cuplikan tutorial Indonesia. |

IndoSum memiliki kolom `id`, `text`, dan `summary`. Skrip meratakan token, merapikan spasi, dan menyaring pasangan kosong atau duplikat eksak. Jika artikel atau ID muncul pada lebih dari satu split, prioritasnya `test`, lalu `validation`, lalu `train`.

Di manifest FLEURS, `audio_path` relatif terhadap folder CSV. Di manifest LUMINA, `video_path` juga relatif terhadap folder CSV yang bersangkutan. Klip LUMINA menyatukan video dengan WAV resmi berdasarkan ID klip. Teks referensi diambil dari Excel resmi berdasarkan nomor kalimat. Jika audio lebih panjang, frame terakhir video ditahan hingga audio selesai. Klip hasilnya bisa langsung diunggah ke Streamlit.

IndoSum dipakai untuk eksperimen berita. FLEURS, LUMINA, dan NSS-ID mendukung evaluasi ucapan bahasa Indonesia. Pengujian kualitas ringkasan pada video Python alami memerlukan referensi khusus dari tutorial itu. Dataset Mozilla yang dibahas sebelumnya mensyaratkan permintaan akses. [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu1ieso2001bo107h8377g5v)

## Muat data peringkasan

```python
from datasets import load_dataset

corpus = "indosum"
splits = ("train", "validation", "test")
dataset = load_dataset("csv", data_files={
    split: f"data/processed/{corpus}/{split}.csv" for split in splits
})
```

Gunakan `text` sebagai masukan mBART dan `summary` sebagai target. Untuk Whisper, gunakan media yang dirujuk manifest dan `transcript` sebagai referensi. Normalisasikan referensi dan prediksi dengan aturan yang sama sebelum menghitung WER atau CER.

## Pilot coding dan NSS-ID

Pilot coding mencakup perulangan, percabangan, fungsi, pandas, dan Boolean. Keluaran contoh numerik dibangun dari aturan Python. Anggap skor pilot sebagai hasil pada skenario sintetis. Gunakan referensi dari tutorial alami terpisah untuk mengukur kinerja pada video nyata.

```powershell
& $projectPython -m scripts.prepare_coding_pilot
```

NSS-ID memakai rekaman penutur manusia dari kategori Imperative dan split uji resmi. Sampelnya mengukur pengenalan ucapan umum. Untuk uji istilah coding, cocokkan keluaran Whisper dengan narasi dari tutorial beranotasi. [Kartu NSS-ID dan lisensi](https://huggingface.co/datasets/Atika88/Indonesian-ASR-11-Class-Dataset) · [Artikel dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC13634895/)

Langkah menyiapkan split latih NSS-ID dan klip audio dua video coding ada di [panduan notebook](../notebooks/README.md#siapkan-data).

```powershell
New-Item -ItemType Directory -Force data/raw/nss-id
$nss = 'https://huggingface.co/datasets/Atika88/Indonesian-ASR-11-Class-Dataset/resolve/97a862f3f1315649a3520bb88d8dc424be96fbcf'
curl.exe -fL "$nss/viewer/metadata/test.parquet" -o data/raw/nss-id/test.parquet
curl.exe -fL "$nss/viewer/metadata/val.parquet" -o data/raw/nss-id/val.parquet
curl.exe -fL "$nss/data/audio_shards/by_category/Imperative.tar" -o data/raw/nss-id/Imperative.tar
& $projectPython -m scripts.prepare_nss_id
$env:WHISPER_MODEL_ID = 'small'
& $projectPython -m scripts.evaluate_media whisper data/processed/nss-id/test.csv --limit 24 --output outputs/nss-id-small.json
```

## Percobaan pada video Python nyata

`samples/coding_video_segments.json` menyimpan ID YouTube, interval, ringkasan acuan, dan detail yang perlu muncul kembali. Isi transkrip diperoleh dari subtitle dan disimpan lokal di `data/raw/`. Semua cuplikan dari satu video masuk satu split. Pembagiannya adalah 18 cuplikan dari 6 video untuk latih, 6 cuplikan dari 3 video untuk validasi, dan 9 cuplikan dari 3 video untuk uji. Data latih juga memuat 28 contoh sintetis dan 10 cuplikan CrossTask. Validasi serta uji hanya memakai video Python Indonesia.

Cuplikan CrossTask memasangkan label tindakan dari arsip resmi dengan subtitle Inggris dari video yang masih tersedia. Label ini lebih singkat daripada ringkasan tutorial. Skrip mengecek kata utama label muncul di sekitar interval anotasi dan memberi tag `en_XX`; contoh Python memakai `id_ID`. [Repositori dan arsip CrossTask](https://github.com/DmZhukov/CrossTask#data)

How2 dan GUIDE belum masuk ke percobaan ini. How2 mengumumkan bahwa banyak video sudah hilang dan unduhan resmi tidak lagi menghasilkan salinan korpus yang berguna. Tautan unduhan GUIDE di Baidu memberikan HTTP 403 pada percobaan ini. [Pengumuman How2](https://github.com/srvk/how2-dataset#important-notice) · [Halaman GUIDE](https://guide-ijcai2024.github.io/)

```powershell
& $projectPython -m scripts.prepare_coding_pilot
& $projectPython -m scripts.prepare_video_experiment
& $gpuPython -m scripts.train_mbart_pilot --data-dir data/processed/coding_video_mix --output-dir outputs/mbart-coding-video-mix --source-tokens 256 --target-tokens 72 --train-samples 56 --validation-samples 6 --decoder-layers 2 --epochs 2 --learning-rate 0.0001 --compare-base
& $gpuPython -m scripts.evaluate_mbart outputs/mbart-coding-video-mix --data-dir data/processed/coding_video_mix --split test --output outputs/coding-video-mix-test.json
```

## Sampel kode multibahasa

Sampel The Vault dibuat dari shard fungsi rilis `505c679056e49a2a269b64777ee7c496d22e1440`. Pilihan data latih mencakup variabel, kondisi, perulangan, array atau list, serta fungsi pada setiap bahasa. Label topik dideteksi dari konstruksi kode. Setiap fungsi dipasangkan dengan deskripsi docstring Inggris. Repositori asal, revisi, dan lisensi disimpan di CSV, dengan data uji dari repositori terpisah. Sampel ini mengukur ringkasan fungsi. Kualitas ringkasan narasi Indonesia diukur pada split video terpisah. [The Vault Function](https://huggingface.co/datasets/Fsoft-AIC/the-vault-function)

```powershell
& $projectPython -m scripts.prepare_multilingual_code --download
```

## Siapkan ulang di komputer lain

```powershell
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements-training.txt
New-Item -ItemType Directory -Force data/raw/indosum,data/raw/fleurs
curl.exe -fL 'https://drive.usercontent.google.com/download?id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco&export=download&confirm=t' -o data/raw/indosum/indosum.tar.gz
curl.exe -fL 'https://huggingface.co/datasets/google/fleurs/resolve/94d6bc3ce99786003b4c2432bee7f87bf0732e77/id_id/test-00000-of-00001.parquet' -o data/raw/fleurs/test.parquet
& $projectPython -m scripts.fetch_lumina
& $projectPython -m scripts.prepare_datasets
```

## Sumber dan ketentuan pakai

- **IndoSum.** Gunakan sitasi Kurniawan dan Louvan (2018). README di dalam arsip korpus menyatakan CC BY-SA 4.0. Repo kode mencantumkan Apache 2.0. [Repo resmi](https://github.com/kata-ai/indosum) · [Paper](https://arxiv.org/abs/1810.05334)
- **FLEURS.** CC BY 4.0. Gunakan atribusi pembuat dataset. [Kartu dataset Google](https://huggingface.co/datasets/google/fleurs)
- **LUMINA v4.** CC BY 4.0. Gunakan atribusi Setyaningsih, Chen, Kristian, Handayani, dan Irianto (2024) serta sebutkan bahwa WAV dipasang pada video untuk paket pengujian lokal. [Mendeley Data](https://data.mendeley.com/datasets/8fw93k4rny/4)
