# Panduan Praktik Python

Aplikasi lokal yang merangkum video tutorial Python berbahasa Indonesia menjadi langkah praktik dengan penanda waktu. Unggah MP4 bernarasi, tekan **Buat Panduan**, lalu pilih waktu pada langkah yang ingin diputar. Gunakan **Lihat transkrip** untuk membaca teks sumber. Hasil tersedia selama sesi browser. **Ganti video** membuka tampilan unggah kembali.

## Jalankan di Windows PowerShell

Pasang Python 3.11, FFmpeg, dan FFprobe. Buka ulang PowerShell setelah pemasangan agar `ffmpeg` dan `ffprobe` ditemukan di `Path`. Untuk panduan prasyarat, lihat [SETUP.md](SETUP.md).

```powershell
py -3.11 -m venv .venv
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
& $projectPython -m pip install -r requirements.txt
& $projectPython -m pip check
& $projectPython -m streamlit run app.py --server.address 127.0.0.1
```

Jika Python 3.11 dipasang melalui `uv`, gunakan `uv venv --python 3.11 --seed .venv` untuk langkah pertama. Buka alamat lokal yang ditampilkan Streamlit. Uji awal sebaiknya memakai video pendek karena Whisper dan IndoBART-v2 berjalan di CPU. Sambungan internet diperlukan untuk mengunduh model pertama kali.

Jalankan `& $projectPython -m unittest discover -s tests -v` untuk menguji batas segmen dan penjagaan detail ringkasan. Lingkungan uji Windows menggunakan Python 3.11.15, FFmpeg 9.1, Streamlit 1.67.0, PyTorch 2.15.0 CPU, dan Transformers 4.57.6.

Uji Chrome dengan klip demonstrasi pandas bernarasi sintetis menghasilkan tiga langkah pada `00:00`, `00:10`, dan `00:19`. Klik setiap waktu, klik ulang waktu yang sama, panel transkrip, tata letak lebar dan sempit, serta pergantian unggahan sudah diuji. Uji dengan tutorial publik berizin masih tertunda karena izin unduh kandidat belum terverifikasi.

## Model dan batasan

- Whisper `base` multibahasa mentranskripsi audio sebagai bahasa Indonesia. Waktu awal dan akhir langkah berasal dari segmen Whisper. Pemutar Streamlit memakai detik utuh.
- Model bawaan `indobenchmark/indobart-v2` adalah checkpoint publik pralatih. Uji CPU memakai revisi `7192ee75ba70ca247c7abfb8e7268588145c0bde`. Untuk memakai bobot hasil fine-tuning, atur `$env:INDOBART_CHECKPOINT` ke direktori checkpoint Transformers yang kompatibel sebelum menjalankan aplikasi. Gunakan tokenizer IndoNLG yang cocok dengan checkpoint. Panduan menampilkan rujukan model yang benar-benar dipakai.
- Batas langkah diperkirakan dari peralihan aksi, jeda, dan panjang teks. Jika ringkasan kosong atau pemeriksaan mendeteksi perubahan detail, langkah menampilkan **Cuplikan sumber**. Nama fungsi yang tidak terdengar dalam narasi atau keliru ditranskripsi perlu diverifikasi melalui demonstrasi video.

## Protokol evaluasi

1. Gunakan [IndoSum](https://github.com/kata-ai/indosum) untuk eksperimen awal peringkasan teks bahasa Indonesia. Dependensi riset tersedia di `requirements-training.txt`.
2. Untuk evaluasi tutorial, kumpulkan MP4 tutorial berbahasa Indonesia. Catat `video_id`, sumber, pembuat, dan lisensi. Untuk tiap langkah, anotasi ringkasan referensi, istilah teknis penting, `start`, dan `end` demonstrasi. Simpan berkas media uji di `samples/local/`.
3. Pisahkan data latih dan uji menurut `video_id` sebelum membuat potongan transkrip. Catat checkpoint yang dipakai pada setiap hasil. Nilai cakupan langkah penting, ketepatan fungsi dan parameter, serta persentase penanda waktu yang membuka rentang demonstrasi yang benar. Laporkan langkah yang memakai cuplikan sumber.
4. Bandingkan panduan dan transkrip biasa dengan tugas pencarian instruksi yang sama. Acak urutan kondisi, ukur waktu hingga peserta menemukan demonstrasi yang tepat, dan catat persentase keberhasilan pada tiap kondisi.
