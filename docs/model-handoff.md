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

```powershell
$cpuPython = '.\.venv\Scripts\python.exe'
& $cpuPython -m pip install -r requirements-training.txt
$env:MBART_MODEL_ID = 'outputs/mbart-300-layer2/adapter'
& $cpuPython -m scripts.check_mbart
& $cpuPython -m streamlit run app.py
```

Jalur pemrosesan panduan berhasil memuat adapter dua layer dengan konfigurasi di atas. Kirim folder `adapter` bersama `metrics.json` dan revisi model dasar ke tim aplikasi. Gunakan `Remove-Item Env:MBART_MODEL_ID` untuk memakai konfigurasi bawaan Streamlit.
