# Eksperimen dan serah terima model

Model dasar adalah `facebook/mbart-large-50@4ef55a20b36c6903b832e38f0604ab4bdf22c7d6` dengan tokenizer `id_ID`. Aplikasi mengambil hasil Whisper `base` per segmen, mengelompokkan tindakan, lalu menggunakan mBART untuk menyiapkan ringkasan langkah.

## Eksperimen IndoSum

Gunakan fold resmi IndoSum `01` yang sudah disiapkan. Semua percobaan konteks panjang memakai 20 sampel validasi yang sama, batas masukan 1.024 token dan keluaran 128 token, seed 42, batch size 1, dan satu epoch. LoRA memakai rank 8, alpha 16, dropout 0,05, dan AdamW dengan learning rate `2e-5`. Opsi `--decoder-layers` memilih berapa layer terakhir decoder yang diberi LoRA pada proyeksi query dan value.

```powershell
$gpuPython = '.\.venv-rocm\Scripts\python.exe'
& $gpuPython -m scripts.train_mbart_pilot --train-samples 300 --decoder-layers 1 --source-tokens 1024 --target-tokens 128 --output-dir outputs/mbart-300-layer1
& $gpuPython -m scripts.train_mbart_pilot --train-samples 300 --decoder-layers 2 --source-tokens 1024 --target-tokens 128 --output-dir outputs/mbart-300-layer2
```

Gunakan folder keluaran baru saat mengulang perintah. Setiap folder berisi `metrics.json`, `predictions.json`, dan subfolder `adapter`. Pembanding model dasar pada sampel validasi yang sama tersedia di `outputs/mbart-pilot-1024-128/baseline_metrics.json`. Split `test` disimpan untuk evaluasi akhir setelah model dipilih.

| Konfigurasi | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| mBART dasar | 67,77 | 59,38 | 65,21 |
| 100 contoh, LoRA 1 layer | 67,74 | 59,35 | 65,19 |
| 300 contoh, LoRA 1 layer | 67,69 | 59,31 | 65,07 |
| 300 contoh, LoRA 2 layer | 67,69 | 59,31 | 65,07 |

Sampel 300 contoh dimulai dengan 100 contoh dari percobaan sebelumnya dalam urutan yang sama. ID latih dan validasi tidak beririsan. Untuk aplikasi, gunakan checkpoint dasar karena skornya tertinggi pada validasi ini. Kedua adapter 300 contoh dapat dipakai untuk eksperimen lanjutan. Skor di atas memakai 20 artikel IndoSum dan belum mengukur kualitas pada tutorial Python alami.

```powershell
& $gpuPython -m scripts.evaluate_mbart outputs/mbart-300-layer2 --split test --output outputs/indosum-final-test.json
```

Tanpa `--limit`, perintah mengevaluasi seluruh split dua kali untuk model dasar dan adapter. Gunakan folder eksperimen yang dipilih pada evaluasi akhir.

## Eksperimen 1.000 artikel dengan maksimal 5 epoch

Gunakan 100 artikel validasi yang sama untuk semua epoch dan pembanding. AdamW melanjutkan status optimizer antar-epoch. Training berhenti setelah dua epoch berturut-turut tanpa kenaikan ROUGE-L dan menyimpan adapter dari epoch dengan ROUGE-L tertinggi. Batas token tetap `1024/128`.

```powershell
& $gpuPython -u -m scripts.train_mbart_pilot --train-samples 1000 --validation-samples 100 --decoder-layers 2 --source-tokens 1024 --target-tokens 128 --epochs 5 --patience 2 --compare-base --output-dir outputs/mbart-indo1000-max5-ulang
```

Lihat `epoch_history.json` untuk loss dan ROUGE tiap epoch, serta `metrics.json` untuk skor epoch terbaik. Dengan `--compare-base`, setelah training skrip juga menghitung skor mBART dasar dan Lead-3 pada 100 artikel yang sama. Lead-3 memakai tiga kalimat pembuka yang dipisahkan berdasarkan tanda akhir kalimat.

## Evaluasi ucapan dan panduan

```powershell
$cpuPython = '.\.venv\Scripts\python.exe'
& $cpuPython -m scripts.evaluate_media whisper data/processed/fleurs/test.csv --limit 8 --output outputs/fleurs-whisper.json
& $cpuPython -m scripts.evaluate_media whisper data/processed/lumina/test.csv --limit 8 --output outputs/lumina-whisper.json
& $cpuPython -m scripts.evaluate_media tutorial outputs/demo-pandas-narasi-sintetis.mp4 samples/pandas_synthetic_reference.json --output outputs/pandas-smoke.json
```

WER dan CER dihitung terhadap teks referensi setelah menyamakan huruf besar, spasi, dan tanda baca. Uji pandas memakai narasi dari skrip demo dan waktu pergantian adegan dari FFprobe. Pencocokan langkah memerlukan minimal 50% irisan interval dibandingkan gabungan interval. Timestamp memakai detik bulat sesuai pemutar Streamlit dan harus berada di dalam adegan yang cocok. Istilah dinilai sebagai kecocokan frasa setelah normalisasi yang sama. Klip demo bersifat sintetis. Tutorial alami dengan anotasi langkah, ringkasan, dan waktu sumber masih diperlukan untuk penilaian domain.

| Uji Whisper base | Sampel | WER | CER |
| --- | ---: | ---: | ---: |
| FLEURS Indonesia | 8 | 38,64% | 13,85% |
| LUMINA v4 | 8 | 60,42% | 21,84% |
| Klip pandas sintetis | 1 | 31,11% | 6,85% |

Pada klip pandas, kedua model menghasilkan tiga langkah dan timestamp yang masuk ke adegan referensi pada `00:00`, `00:10`, dan `00:19`. Keduanya mempertahankan 2 dari 6 frasa teknis secara persis. Di langkah `00:10`, referensi ringkasannya memakai `dropna` pada kolom `harga`, sedangkan ringkasan aktual mengikuti teks Whisper yang berbunyi “fungsi drop na dan pilih olom harga”. Laporan dan contoh lengkap ada di `outputs/pandas-smoke.json` dan `outputs/pandas-smoke-adapter.json`.

## Muat adapter di aplikasi

### Pilot tutorial coding dan pembanding Whisper

Adapter `outputs/mbart-coding-pilot/adapter` dilatih satu epoch dengan 56 contoh sintetis orisinal. LoRA diterapkan pada dua lapisan terakhir decoder dengan learning rate `0.0001` dan batas token `128/56`. Validasi memakai 24 contoh dan uji memakai 28 contoh dari skenario terpisah.

| Model mBART pada 28 contoh uji | ROUGE-1 | ROUGE-2 | ROUGE-L | Recall istilah teknis dan angka |
| --- | ---: | ---: | ---: | ---: |
| `facebook/mbart-large-50` | 39,42% | 18,35% | 31,57% | 92,86% |
| Adapter coding | 37,74% | 17,76% | 30,23% | 85,71% |

Recall di atas menghitung istilah teknis dan angka yang muncul kembali secara literal. Ukuran ini tidak memeriksa apakah angka dan hasil operasi dipasangkan dengan tepat. Model dasar dipilih untuk tampilan aplikasi karena hasil pilot lebih baik pada contoh uji ini.

| Whisper CPU INT8 pada 24 klip manusia NSS-ID Imperative | WER | CER |
| --- | ---: | ---: |
| `base` | 47,22% | 13,31% |
| `small` | 27,78% | 7,74% |

Klip berasal dari tiga penutur uji dan berisi ucapan umum. `small` dipakai untuk uji aplikasi lokal. Untuk menggunakan checkpoint yang sama pada sesi lain, atur `$env:WHISPER_MODEL_ID = 'data/raw/models/faster-whisper-small'` jika foldernya tersedia, atau gunakan `small` agar Faster Whisper mengambil checkpoint. Berkas perbandingan tersedia di `outputs/coding-pilot-test.json`, `outputs/nss-id-whisper-base.json`, dan `outputs/nss-id-whisper-small.json`.

```powershell
& $projectPython -m scripts.prepare_coding_pilot
& $gpuPython -m scripts.train_mbart_pilot --data-dir data/processed/coding_pilot --output-dir outputs/mbart-coding-pilot --source-tokens 128 --target-tokens 56 --train-samples 56 --validation-samples 24 --decoder-layers 2 --epochs 1 --learning-rate 0.0001 --compare-base
& $gpuPython -m scripts.evaluate_mbart outputs/mbart-coding-pilot --data-dir data/processed/coding_pilot --split test --output outputs/coding-pilot-test.json
```

### Percobaan CrossTask dan tutorial nyata

`outputs/mbart-coding-video-mix/adapter` dibangun dari `facebook/mbart-large-50` dengan LoRA pada dua lapisan terakhir decoder. Training berlangsung dua epoch pada 56 contoh. Data latih berisi 18 cuplikan dari 6 video Python, 28 pasangan sintetis, dan 10 pasangan subtitle Inggris serta label tindakan CrossTask. Validasi berisi 6 cuplikan dari 3 video Python. Uji berisi 9 cuplikan dari 3 video Python lain tentang list dan fungsi, percabangan, serta set. Identitas video diperiksa agar tidak beririsan antarsplit. [Panduan dataset](../data/README.md#percobaan-pada-video-python-nyata)

| Pengukuran | mBART dasar | Adapter epoch 2 |
| --- | ---: | ---: |
| ROUGE-L F1 validasi | 8,36% | 8,96% |
| ROUGE-1 F1 uji | 19,42% | 15,88% |
| ROUGE-2 F1 uji | 4,08% | 1,09% |
| ROUGE-L F1 uji | 14,45% | 11,12% |
| Detail acuan yang muncul secara literal pada uji | 18/23 | 17/23 |

Metrik uji dihitung dari teks mBART yang dihasilkan langsung dari cuplikan subtitle. Pemeriksaan kecocokan literal mengukur kemunculan istilah atau angka. Pemeriksaan ini tidak menilai hubungan makna antardetail. Dalam satu contoh, adapter menyebut `else` tetapi kehilangan hasil berupa huruf `C`. Hasil ini menjadi alasan untuk memakai mBART dasar di aplikasi. Laporan lengkap tersimpan di `outputs/coding-video-mix-test.json`. Ukuran uji yang kecil membatasi kesimpulan tentang pengaruh CrossTask dan generalisasi ke tutorial lain.

Untuk menguji adapter ini di aplikasi, arahkan `MBART_MODEL_ID` ke `outputs/mbart-coding-video-mix/adapter` setelah memasang dependensi training.

### Percobaan The Vault dan tutorial multibahasa

`outputs/mbart-coding-multilingual/adapter` dilatih pada 106 contoh. Tambahan The Vault berisi 50 pasangan fungsi dan docstring dari 10 bahasa, dengan contoh variabel, percabangan, perulangan, array atau list, serta fungsi. Enam cuplikan Indonesia tetap dipakai untuk validasi. Model mempertahankan 3 dari 13 detail acuan pada epoch 1 dan 2 dari 13 pada epoch 2. Dengan batas 3 epoch dan patience 1, training berhenti seusai epoch 2 dan menyimpan epoch 1.

| Pengukuran | mBART dasar | Adapter multibahasa |
| --- | ---: | ---: |
| ROUGE-L F1 pada 9 cuplikan uji video Python | 14,34% | 12,49% |
| Detail muncul secara literal pada uji video | 15/23 | 14/23 |
| ROUGE-L F1 pada 20 fungsi uji The Vault | 8,42% | 8,42% |

Hasil uji ini mendukung pemakaian mBART dasar di aplikasi. Laporan mentah tersimpan di `outputs/coding-multilingual-video-test.json` dan `outputs/coding-multilingual-vault-test.json`. Uji video menggunakan kembali cuplikan regresi dari percobaan sebelumnya. Uji The Vault memakai repositori yang berbeda dari sampel latih.

```powershell
& $projectPython -m scripts.prepare_multilingual_code --download
& $gpuPython -m scripts.train_mbart_pilot --data-dir data/processed/coding_multilingual_mix --output-dir outputs/mbart-coding-multilingual --source-tokens 256 --target-tokens 48 --train-samples 106 --validation-samples 6 --decoder-layers 2 --epochs 3 --patience 1 --learning-rate 0.0001 --select-by details --compare-base
& $gpuPython -m scripts.evaluate_mbart outputs/mbart-coding-multilingual --data-dir data/processed/coding_multilingual_mix --split test --output outputs/coding-multilingual-video-test.json
& $gpuPython -m scripts.evaluate_mbart outputs/mbart-coding-multilingual --data-dir data/processed/multilingual_code --split test --output outputs/coding-multilingual-vault-test.json
```

### Pemilihan checkpoint dan beam Whisper

Empat konfigurasi Faster Whisper CPU INT8 divalidasi pada 12 klip dari tiga penutur NSS-ID. WER validasi `small` adalah 36,99% dengan satu beam dan 35,62% dengan tiga beam. Kedua konfigurasi Turbo mencapai 21,92% WER dan 3,80% CER. Tiga beam dipilih berdasarkan waktu pada pengukuran validasi ini. Evaluasi akhir memakai 24 klip dari tiga penutur pada split uji resmi.

| Konfigurasi pada split uji NSS-ID | WER | CER | Waktu total transkripsi |
| --- | ---: | ---: | ---: |
| `small`, beam 1 | 27,78% | 7,74% | 57,1 detik |
| `small`, beam 3 | 21,53% | 6,81% | 56,5 detik |
| `large-v3-turbo`, beam 3 | 9,03% | 2,27% | 205,8 detik |

Pada satu klip pandas sintetis, WER `small` beam 1 adalah 24,44% dan Turbo beam 3 mencapai 13,33%. Turbo masih membaca `dropna` sebagai dua kata dan melewatkan penyebutan CSV pada beberapa bagian. Aplikasi lokal dipasang dengan Turbo beam 3 untuk mengutamakan akurasi. Eksperimen mengubah checkpoint dan cara decoding, dengan bobot Whisper resmi dari checkpoint. Laporan tersedia di `outputs/whisper-beam-small.json` dan `outputs/whisper-decoding-pilot.json`.

```powershell
& $projectPython -m scripts.prepare_nss_id --split val --per-speaker 4
& $projectPython -m scripts.tune_whisper --small data/raw/models/faster-whisper-small --turbo data/raw/models/faster-whisper-turbo
```

### Perintah pemuatan adapter

```powershell
$cpuPython = '.\.venv\Scripts\python.exe'
& $cpuPython -m pip install -r requirements-training.txt
$env:MBART_MODEL_ID = 'outputs/mbart-300-layer2/adapter'
& $cpuPython -m scripts.check_mbart
& $cpuPython -m streamlit run app.py
```

Jalur pemrosesan panduan berhasil memuat adapter dua layer dengan konfigurasi di atas. Kirim folder `adapter` bersama `metrics.json` dan revisi model dasar ke tim aplikasi. Gunakan `Remove-Item Env:MBART_MODEL_ID` untuk memakai konfigurasi bawaan Streamlit.
