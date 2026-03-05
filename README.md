# SIBI Hand Gesture Recognition

Aplikasi pengenalan gesture tangan BISINDO (Bahasa Isyarat Indonesia) menggunakan teknologi MediaPipe dan Machine Learning. Aplikasi ini dapat mendeteksi 26 huruf A-Z dari gerakan tangan secara real-time.

## 📋 Daftar Isi

- [Fitur](#fitur)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Struktur Proyek](#struktur-proyek)

---

## ✨ Fitur

- ✅ Deteksi hand landmarks menggunakan MediaPipe
- ✅ Pengenalan 26 huruf A-Z
- ✅ Model Machine Learning (Neural Network, Random Forest, Gradient Boosting)
- ✅ Real-time detection via webcam
- ✅ **🔊 Suara untuk setiap huruf yang terdeteksi** (Text-to-Speech)
- ✅ Web interface dengan Flask
- ✅ Pengumpulan data training otomatis
- ✅ Training & evaluasi model

---

## 💻 Persyaratan Sistem

- **Python**: 3.8 atau lebih tinggi
- **OS**: Windows, macOS, atau Linux
- **Webcam**: Required untuk deteksi real-time
- **RAM**: Minimal 4GB

---

## 🔧 Instalasi

### 1. Clone atau Download Proyek

```bash
# Jika menggunakan git
git clone <repository-url>
cd sibi_python

# Atau extract folder proyek jika sudah di-download
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Atau install manual:**

```bash
pip install flask
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install mediapipe
pip install pyttsx3
```

### 4. Verifikasi Instalasi

```bash
python cekinstall.py
```

Jika berhasil, akan muncul:
```
✅ Semua library berhasil diinstall!
OpenCV: 4.x.x
MediaPipe: 0.x.x
Scikit-learn: 1.x.x
```

---

## 📌 Penggunaan

### 1. Mengumpulkan Data Training

Jika Anda ingin melatih model dengan data sendiri:

```bash
python kumpul_data.py
```

**Instruksi:**
- Aplikasi akan meminta Anda menampilkan gesture untuk setiap huruf (A-Z)
- **SPASI**: Mulai merekam
- **Q**: Skip huruf ini
- Akan merekam 200 sampel per huruf secara otomatis
- Data disimpan di folder `data_gesture/`

> ℹ️ Data sudah tersedia di folder `data_gesture/`, Anda bisa skip langkah ini jika tidak perlu melatih ulang

### 2. Melatih Model

Untuk melatih atau melatih ulang model dengan data terbaru:

```bash
python latih_model.py
```

**Proses:**
- Membaca semua file `.npy` dari `data_gesture/`
- Melakukan normalisasi landmark
- Training dengan multiple models (Neural Network, Random Forest, Gradient Boosting)
- Voting ensemble untuk prediksi akhir
- Menyimpan model ke file pickle:
  - `model_bisindo_nn.pkl`
  - `label_encoder.pkl`
  - `scaler.pkl`

**Output:**
- Accuracy score pada test data
- Classification report per huruf

### 2. Deteksi Real-Time (Command Line)

Untuk menjalankan deteksi langsung via terminal:

```bash
python deteksi.py
```

**Instruksi:**
- Tampilkan gesture tangan di depan webcam
- Sistem akan menampilkan prediksi huruf secara real-time
- **🔊 Setiap huruf yang terdeteksi akan diucapkan secara otomatis**
- Tekan **Q** untuk keluar

**Fitur:**
- Menampilkan landmark tangan (21 titik)
- Confidence score untuk setiap prediksi
- Smooth prediction dengan window history (10 frame)
- Delay 2 detik sebelum menambah huruf baru
- **Text-to-Speech otomatis untuk feedback audio**

### 4. Aplikasi Web (Flask)

Untuk menggunakan interface web dengan fitur lebih lengkap:

```bash
python app.py
```

**Akses aplikasi:**
- Buka browser: `http://localhost:5000`

**Fitur Web:**
- 📹 Live webcam streaming
- 🔤 Real-time gesture recognition
- � **Audio feedback untuk setiap huruf terdeteksi**
- 📝 Construct sentences dari gestures
- ➕ Accumulate words/letters
- 🗑️ Clear/Reset functionality
- 🎯 Visual feedback dengan bounding boxes

**Cara Menggunakan Web Interface:**
1. Buka `http://localhost:5000` di browser
2. Izinkan akses webcam
3. Tampilkan gesture huruf di depan kamera
4. **🔊 Huruf akan diucapkan otomatis saat terdeteksi**
5. Huruf ditampilkan di layar secara real-time
6. Tekan tombol untuk mengakumulasi kata atau kalimat

---

## 📁 Struktur Proyek

```
sibi_python/
├── app.py                          # Flask web application
├── deteksi.py                      # Real-time detection (terminal)
├── kumpul_data.py                  # Data collection script
├── konversi_dataset.py             # Dataset conversion utility
├── latih_model.py                  # Model training script
├── cekinstall.py                   # Dependency checker
├── tts_handler.py                  # Text-to-Speech handler
│
├── hand_landmarker.task            # MediaPipe pre-trained model
│
├── model_bisindo_nn.pkl            # Trained model (Neural Network)
├── label_encoder.pkl               # Label encoder
├── scaler.pkl                      # Feature scaler
│
├── data_gesture/                   # Training data
│   ├── A/ (200 samples per letter)
│   ├── B/
│   ├── ...
│   └── Z/
│
├── data_gambar/                    # Additional image data
├── datagesture/                    # Converted gesture data
│
├── templates/
│   └── index.html                  # Web interface HTML
│
├── requirements.txt                # Python dependencies
└── README.md                       # Dokumentasi ini
```

---

## 🔍 Detail File

| File | Fungsi |
|------|--------|
| `app.py` | Flask server untuk web interface dengan live streaming & suara otomatis |
| `deteksi.py` | Terminal-based real-time gesture detection dengan TTS |
| `kumpul_data.py` | Mengumpulkan data training dari webcam |
| `latih_model.py` | Training model menggunakan sklearn ensemble |
| `konversi_dataset.py` | Utility untuk konversi format data |
| `cekinstall.py` | Memeriksa instalasi library yang diperlukan |
| `tts_handler.py` | Handler untuk Text-to-Speech (optional untuk implementasi lanjut) |
| `hand_landmarker.task` | Model MediaPipe untuk landmark detection |
| `requirements.txt` | Daftar semua dependencies Python |

---

## 🎯 Workflow Umum

### Workflow A: Menggunakan Model yang Sudah Ada
```bash
# 1. Verifikasi instalasi
python cekinstall.py

# 2. Jalankan web app atau deteksi terminal
python app.py          # Atau python deteksi.py
```

### Workflow B: Training Model Baru dengan Data Sendiri
```bash
# 1. Verifikasi instalasi
python cekinstall.py

# 2. Kumpulkan data
python kumpul_data.py

# 3. Train model
python latih_model.py

# 4. Jalankan deteksi
python app.py          # Atau python deteksi.py
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"
```bash
pip install flask
```

### Error: "ModuleNotFoundError: No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Webcam tidak terdeteksi
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Ubah device index pada line `cap = cv2.VideoCapture(0)` menjadi `1`, `2`, dst.

### Port 5000 sudah digunakan
Edit `app.py` dan ubah port:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Ganti port menjadi 8000
```

### Model accuracy rendah
- Kumpulkan lebih banyak data per huruf
- Pastikan gesture dilakukan dengan jelas di depan kamera
- Training ulang dengan `python latih_model.py`

---

## 📊 Model Architecture

Proyek ini menggunakan **Voting Classifier** dengan 3 models:
1. **Neural Network (MLPClassifier)** - Flexible, good generalization
2. **Random Forest** - Robust, handles non-linear patterns
3. **Gradient Boosting** - High accuracy, sequential learning

Ensemble voting mengambil prediksi mayoritas dari ketiga model untuk hasil yang lebih akurat.

## 🔊 Fitur Text-to-Speech

Aplikasi dilengkapi dengan fitur suara otomatis yang menggunakan **pyttsx3**:

- **Real-time Audio Feedback**: Setiap huruf yang terdeteksi langsung diucapkan
- **Offline**: Tidak butuh koneksi internet
- **Customizable**: Kecepatan dan volume bisa disesuaikan di file `deteksi.py` atau `app.py`
- **Non-blocking**: Suara tidak mengganggu proses deteksi

Untuk mengatur kecepatan suara, edit file:

```python
tts_engine.setProperty('rate', 150)      # 100-200 (semakin besar = semakin cepat)
tts_engine.setProperty('volume', 1.0)    # 0.0 - 1.0 (0 = senyap, 1 = maksimal)
```

---

## 🚀 Tips & Trik

1. **Lighting**: Pastikan pencahayaan cukup untuk deteksi tangan yang optimal
2. **Background**: Gunakan background yang kontras dengan tangan
3. **Distance**: Posisikan tangan sekitar 30-50cm dari kamera
4. **Speed**: Gesture dilakukan dengan kecepatan normal, tidak terlalu cepat
5. **Data Collection**: Kumpulkan data dari berbagai angle dan jarak

---

## 📝 Notes

- Model pre-trained `hand_landmarker.task` sudah disediakan dalam proyek
- Dataset training sudah tersedia di `data_gesture/` (200+ sampel per huruf)
- Untuk deteksi optimal, pastikan minimal 150-200 sampel per huruf

---

## 📞 Support

Jika ada pertanyaan atau issue, silakan:
- Buat issue di repository
- Check troubleshooting section di README ini

---

**Happy Gesture Recognizing! 🎉**
