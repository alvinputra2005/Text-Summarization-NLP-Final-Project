# Notebook proyek NLP

**File utama untuk tugas** adalah [Pipeline_NLP_Video_Pemrograman.ipynb](Pipeline_NLP_Video_Pemrograman.ipynb). Notebook ini menghubungkan eksperimen Whisper, perbandingan tiga peringkas, dan evaluasi daftar langkah dari video utuh. File individual di bawah menyimpan rincian eksperimen sebelumnya.

Notebook utama terdiri dari 18 tahap, mulai dari eksplorasi data hingga transkrip lengkap dan daftar instruksi. Eksperimen mBART membandingkan bobot dasar, adaptasi langsung pada tutorial, dan adaptasi bertahap IndoSum → tutorial. Tabel mBART, mT5, dan IndoBART generasi pertama memuat hasil pembanding dasar dari arsip. [Registri sumber](../samples/summary_dataset_sources.json)

Whisper memakai AdamW dengan learning rate awal `1e-5`. mBART memakai AdamW dengan learning rate tetap `2e-5` dan memperbarui langsung dua matriks cross-attention. Mode `smoke` memakai dua contoh per split tiap dataset dan satu epoch per tahap. Mode `pilot` memakai sampai lima epoch per tahap dengan early stopping dua epoch. Notebook mencatat jumlah contoh dan checkpoint aktual.

Pasang dependensi notebook pada `.venv` serta interpreter `.venv-rocm` yang akan melatih model jika memakai GPU. Agar kode GPU yang sudah terpasang tetap dipakai, jalankan instalasi tanpa membuat ulang lingkungan PyTorch.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements-notebooks.txt
```

Pengujian video baru menulis `transcripts.csv`, `summary_points.csv` dan laporan JSON. Gunakan `src.summary_api.summarize_csv` untuk CSV masukan dengan kolom `id,text,reference_summary`. API lokal `POST /summarize` menerima satu teks dalam JSON dengan kolom `text` dan `reference_summary` opsional. API memakai checkpoint dari konfigurasi deployment.

Untuk menjalankan demo lokal dengan checkpoint dasar yang dipilih dari validasi coding, arahkan `SUMMARY_DEPLOYMENT` ke `deployment.json` yang ditampilkan notebook.

```powershell
$env:SUMMARY_DEPLOYMENT = 'outputs/notebooks/seq2seq-comparison-v1/deployment.json'
.\.venv\Scripts\python.exe -m uvicorn src.summary_api:app --host 127.0.0.1 --port 8000
```

Dokumentasi API ada di `http://127.0.0.1:8000/docs`. Pada terminal lain, set `SUMMARY_DEPLOYMENT` ke file yang sama dan jalankan `.\.venv\Scripts\python.exe -m streamlit run app.py` jika ingin memilih checkpoint itu untuk video. Streamlit tetap bisa berjalan dengan konfigurasi model bawaannya.

Untuk file utama, pilih kernel `.venv` di VS Code dan klik **Run All**. Mode bawaan `replay` memuat hasil yang disertakan. Atur `NLP_NOTEBOOK_MODE=smoke` atau `pilot` sebelum membuka kernel untuk melatih ulang. mBART memakai CPU secara bawaan. Untuk GPU yang mendukung BF16, isi `NLP_MBART_PYTHON` dengan path interpreter GPU dan `NLP_MBART_DEVICE=cuda:0`.

Atur `NLP_NOTEBOOK_FRESH_VIDEO=1` untuk mengulang inferensi video. Uji video lokal memakai cache Whisper small jika tersedia. Apabila pelatihan ulang dan video baru diaktifkan bersama, notebook memakai checkpoint Whisper dan mBART dari eksekusi tersebut.

Jika Whisper adaptasi tidak mengembalikan segmen atau menghasilkan rentang segmen lebih dari 45 detik, notebook mengulang transkripsi memakai Faster Whisper dan mencatat model yang akhirnya dipakai.

Untuk memuat hasil pelatihan dari eksekusi terdahulu, atur `NLP_NOTEBOOK_WHISPER_CHECKPOINT` ke folder `checkpoint` dan `NLP_NOTEBOOK_MBART_WEIGHTS` ke berkas `selected.pt`. Notebook menerapkan keduanya saat mengulang inferensi video.

## Ringkasan video utuh dan evaluasi multimodal

Bagian 07–10 menelusuri transkripsi, cleaning, OCR, dan Gemini. Bagian 14 serta 17–18 menampilkan inferensi, evaluasi video, dan keluaran akhir. Dua video publik tercatat di [manifest video](../samples/notebook_video_holdout.json). Empat kasus terkontrol tercatat di [manifest regresi preprocessing](../samples/preprocessing_regression.json). Atur `NLP_PREPROCESSING_EVAL=1` untuk mengulang regresi pada bagian 17. Video publik ikut diuji jika cache transkrip tersedia.

Bagian akhir memuat transkrip asli dan bersih, lalu tiga jalur ringkasan. A memakai transkrip, B menambah OCR lokal, dan C menambah bukti Gemini. C memiliki keluaran dan skor jika bukti dari API lolos pemeriksaan. Ketiga jalur menggunakan model dan konfigurasi decoding yang sama dalam satu eksperimen.

**Run All** dalam mode `replay` menampilkan hasil uji lokal dari video Break. Grafik menunjukkan cakupan fakta, detail kode yang tepat, ROUGE, dan durasi pemrosesan. Acuan disimpan di [manifest evaluasi video](../samples/notebook_video_holdout.json). Jalur ini dijalankan melalui `scripts.notebook_video_ablation` di notebook. [Video Break dari Kelas Terbuka](https://www.youtube.com/watch?v=B6scLunzn0I)

Notebook mengukur kontribusi OCR terhadap panduan pada video acuan. Streamlit memakai pemeriksaan transkrip dan frame yang sama untuk MP4 serta YouTube jika Tesseract tersedia. Aplikasi memakai checkpoint mBART dasar secara bawaan dan memberi peringatan ketika kuota Gemini tercapai. [Menjalankan OCR di Streamlit](../README.md#pembacaan-kode-di-layar)

Untuk mengulang uji, siapkan FFmpeg dan Tesseract dengan data bahasa `eng` pada mesin lokal. Jika Tesseract dipasang di lokasi khusus, isi `TESSERACT_EXE` dengan path ke `tesseract.exe`. Link publik dipakai untuk menampilkan video dan mengambil MP4 untuk OCR lokal melalui `yt-dlp`. Siapkan Node.js untuk alur tautan atau atur `NLP_MULTIMODAL_VIDEO` ke MP4 sumber yang sama agar unduhan dilewati.

Untuk mengaktifkan Gemini, siapkan `GEMINI_API_KEY` di `.env`, lalu set dua variabel berikut sebelum membuka kernel VS Code. Set batas permintaan menjadi `0` untuk mengulang uji dengan OCR lokal.

```powershell
$env:NLP_MULTIMODAL_RUN = '1'
$env:NLP_GEMINI_MAX_REQUESTS = '4'
```

Satu permintaan Gemini memuat konteks dari paling banyak tiga bagian dan dua crop editor untuk setiap bagian. Sistem mencocokkan kutipan dengan transkrip dan memeriksa kode terhadap hasil OCR kedua frame. Notebook menampilkan temuan yang lolos, jumlah permintaan, dan token jika dilaporkan oleh API. [Masukan gambar Gemini](https://ai.google.dev/gemini-api/docs/image-understanding) · [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)

Untuk video lain, isi `NLP_MULTIMODAL_YOUTUBE_URL` atau `NLP_MULTIMODAL_VIDEO` dan jalankan ulang bagian video. Pipeline membaca seluruh durasi dan memilih frame secara otomatis. `NLP_MULTIMODAL_ROI` menerima `x1 y1 x2 y2` pada rentang 0 sampai 1 jika area editor perlu dibatasi. `NLP_MULTIMODAL_TRANSCRIPT` menerima JSON transkrip bertanda waktu dari video yang sama. Whisper membuat transkrip bila cache belum tersedia. Bukti OCR lokal yang langsung menyusun langkah mendukung kondisi numerik dan kondisi yang membandingkan penghitung dengan variabel input pada `break`. Skor untuk video baru memerlukan fakta referensi di manifest.

## Pembagian data audio

Pada notebook utama, **Audio coding** memiliki 20 klip latih dari dua video, 12 klip validasi dari video perulangan, dan 12 klip uji dari video percabangan. Split dilakukan per video. WER dan CER Audio coding memakai subtitle otomatis sebagai acuan. **LUMINA** memiliki 24 klip latih, 12 validasi, dan 12 uji, dengan penutur yang terpisah antarsplit. Angka tersebut adalah ukuran sampel lokal sebelum pemilihan subset untuk smoke test.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_lumina_splits
.\.venv\Scripts\python.exe -m scripts.prepare_whisper_video --split train
.\.venv\Scripts\python.exe -m scripts.prepare_whisper_video --split validation
.\.venv\Scripts\python.exe -m scripts.prepare_whisper_video --split test
```

Untuk Audio coding, jalankan perintah hanya pada split yang belum ada. Manifest LUMINA baru disimpan di `data/processed/lumina_split/` dan merujuk klip sumber yang sudah tersedia. [LUMINA v4](https://data.mendeley.com/datasets/8fw93k4rny/4) · [Video validasi Audio coding](https://youtu.be/Z4qfMhx4XzQ) · [Video uji Audio coding](https://youtu.be/I8mpr4AvZiQ)

Jalankan notebook dari workspace repo di VS Code dengan **Run All**. Setiap bagian memiliki sel kode, lalu Markdown untuk menjelaskan cara membaca keluarannya. Grafik dan tabel dibuat dari data dan prediksi pada eksekusi tersebut.

| Notebook | Data | Evaluasi |
| --- | --- | --- |
| [01 · mBART dan IndoSum](01_mbart_indosum.ipynb) | Pasangan artikel dan ringkasan Indonesia dari fold IndoSum yang sudah diproses | Loss validasi dan ROUGE pada mBART dasar, LoRA, dan Lead-3 |
| [02 · mBART dan tutorial pemrograman](02_mbart_tutorial_pemrograman.ipynb) | Coding Video Mix, terdiri atas cuplikan tutorial Python beranotasi, contoh sintetis, dan contoh latih CrossTask | ROUGE, recall detail teknis, dan audit cuplikan berdasarkan video utuh |
| [03 · Whisper dan NSS-ID](03_whisper_nss_id.ipynb) | Klip manusia kategori Imperative dari tiga split resmi yang memisahkan penutur | WER, CER, operasi edit, dan ROC deteksi transkrip berisiko |
| [04 · Whisper dan video pemrograman](04_whisper_video_pemrograman.ipynb) | Dua video coding dengan subtitle otomatis untuk adaptasi. Klip LUMINA dengan transkrip manusia untuk validasi dan uji | WER, CER, dan ROC lintas domain. Tersedia sel untuk uji coding independen jika ada acuannya |

## Lingkungan VS Code

Pasang ekstensi Python dan Jupyter di VS Code. Untuk kedua notebook mBART, pilih **Python Environments → .venv-rocm** atau interpreter PyTorch dengan GPU BF16. Kedua notebook Whisper juga mendukung CPU. FFmpeg harus ada di PATH.

```powershell
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements-notebooks.txt
```

Untuk kernel CPU pada notebook Whisper, jalankan perintah yang sama memakai `.venv`. Trainer mBART juga membutuhkan PEFT, Transformers, SentencePiece, dan ROUGE. Lihat [lingkungan pelatihan proyek](../README.md#pelatihan-dan-evaluasi). Pemuat mBART di repo membaca cache checkpoint `facebook/mbart-large-50` beserta bobot `pytorch_model.bin`. Siapkan cache dengan pengujian mBART proyek bila instalasi masih baru.

Pada notebook eksperimen yang melatih model, `smoke` memakai satu epoch dan sampel kecil. Untuk memakai hingga tiga epoch dan sampel lebih besar, atur `MODE = "pilot"` pada sel konfigurasi atau set `NLP_NOTEBOOK_MODE=pilot` di lingkungan kernel sebelum memulai. Setiap eksekusi menyimpan laporan ke folder baru di `outputs/notebooks/`. Untuk laporan tugas akhir, cantumkan mode dan jumlah contoh yang benar-benar dijalankan.

Di Radeon 840M, jalankan satu notebook mBART per kali dan hentikan kernel sebelumnya setelah selesai. Pada pengujian lokal, proses sempat kehabisan memori saat dua kernel diuji berurutan. Menjalankan notebook kedua dari kernel baru menyelesaikan tahap pemuatan bobot.

## Siapkan data

Dataset IndoSum, Coding Video Mix, dan LUMINA menggunakan berkas siap pakai dari [panduan dataset](../data/README.md). Liputan6 tidak dibutuhkan pada konfigurasi ini. IndoSum menjadi pembanding domain berita dan Coding Video Mix menyediakan target yang lebih dekat dengan tutorial. Data tambahan CrossTask hanya masuk split latih.

Jika manifest NSS-ID belum ada, gunakan langkah unduh di [panduan dataset](../data/README.md#pilot-coding-dan-nss-id) untuk arsip audio, metadata validasi, dan metadata uji. Tambahkan metadata latih dari revisi resmi yang sama.

```powershell
$nss = 'https://huggingface.co/datasets/Atika88/Indonesian-ASR-11-Class-Dataset/resolve/97a862f3f1315649a3520bb88d8dc424be96fbcf'
curl.exe -fL "$nss/viewer/metadata/train.parquet" -o data/raw/nss-id/train.parquet
.\.venv\Scripts\python.exe -m scripts.prepare_nss_id --split train
.\.venv\Scripts\python.exe -m scripts.prepare_nss_id --split val --per-speaker 4
.\.venv\Scripts\python.exe -m scripts.prepare_nss_id --split test
```

Notebook 04 menjalankan skrip persiapan bila manifest audio coding belum ada. Audio dan subtitle tetap disimpan secara lokal. Proses tersebut membutuhkan koneksi, FFmpeg, Node.js, dan `yt-dlp` dari lingkungan aplikasi. Untuk media yang kamu punya izin gunakan dan tidak tersedia di YouTube, kamu juga bisa menyediakan sendiri `data/processed/whisper_video/train.csv` dengan kolom `id,video_id,audio_path,transcript,transcript_source`. Gunakan dua ID video yang berbeda dan klip maksimal 30 detik. Path audio relatif terhadap manifest.

Untuk evaluasi coding yang independen, isi variabel `NLP_WHISPER_VIDEO_TEST_CSV` dengan path manifest tambahan yang memuat kolom `id,video_id,audio_path,transcript`. Gunakan video yang berbeda dari kedua video latih serta transkrip yang diverifikasi manusia. Uji LUMINA mengukur ucapan umum. Skor pada LUMINA belum menjawab akurasi istilah coding pada video baru.

Checkpoint Whisper yang dihasilkan memakai format Transformers. Pemutar Streamlit saat ini juga mendukung backend Faster Whisper yang memakai format CTranslate2. Penempatan checkpoint dalam notebook hanya mengubah eksperimen notebook.
