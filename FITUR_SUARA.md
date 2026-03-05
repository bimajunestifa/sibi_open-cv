# 🔊 Panduan Fitur Suara (Text-to-Speech)

## Ringkas

Fitur suara memungkinkan aplikasi untuk **otomatis mengucapkan setiap huruf** yang berhasil dideteksi. Ini memberikan feedback audio real-time kepada user.

## Implementasi

### 1. Terminal Mode (`deteksi.py`)

```bash
python deteksi.py
```

**Fitur:**
- Setiap huruf yang terdeteksi **langsung diucapkan**
- Blocking call (tunggu audio selesai)
- Sederhana dan langsung

**Audio Flow:**
```
Gesture Detected → Confidence > 70% → Huruf Diucapkan → Next Frame
```

### 2. Web Mode (`app.py`)

```bash
python app.py
```

**Fitur:**
- Suara dimainkan di **thread terpisah**
- Tidak blocking - video tetap smooth
- Dapat handle multiple letters

**Audio Flow:**
```
Gesture Detected → Confidence > 70% → Queue Audio → Continue Detection
                                      ↓
                              TTS Thread: Play Audio
```

## Customization

### Mengubah Kecepatan Suara

Edit `deteksi.py` atau `app.py`:

```python
# Lambat (natural) - 100-150
tts_engine.setProperty('rate', 120)

# Normal - 150
tts_engine.setProperty('rate', 150)

# Cepat - 200+
tts_engine.setProperty('rate', 250)
```

### Mengubah Volume

```python
# Senyap - 0.0
tts_engine.setProperty('volume', 0.0)

# Sedang - 0.5
tts_engine.setProperty('volume', 0.5)

# Maksimal - 1.0
tts_engine.setProperty('volume', 1.0)
```

### Mengganti Engine Suara

```python
# Gunakan sistem TTS lokal (Windows SAPI)
tts_engine = pyttsx3.init('sapi5')

# Atau coba voices lain:
voices = tts_engine.getProperty('voices')
for voice in voices:
    print(voice)
    # Gunakan voice:
    # tts_engine.setProperty('voice', voice.id)
```

## Troubleshooting

### ❌ Tidak Ada Suara

1. **Check volume sistem**
   - Pastikan volume Windows/Mac/Linux tidak di-mute

2. **Check speaker**
   - Pastikan speaker/headphone terhubung dan aktif

3. **Restart aplikasi**
   ```bash
   # Kill aplikasi sebelumnya
   # Jalankan ulang
   python app.py
   ```

### ❌ Suara Terganggu

- Naikkan speech rate:
  ```python
  tts_engine.setProperty('rate', 250)
  ```

### ❌ Error: "No module named pyttsx3"

```bash
pip install pyttsx3
```

## Disable Fitur Suara (Optional)

Jika ingin menonaktifkan suara, comment out baris berikut:

**deteksi.py (line 123):**
```python
# speak_letter(prediksi_stabil)  # Uncomment ini untuk disable
```

**app.py (line 153):**
```python
# speak_letter_async(prediksi_stabil)  # Uncomment ini untuk disable
```

## Advanced: Custom TTS Handler

Untuk implementasi lebih advanced, gunakan `tts_handler.py`:

```python
from tts_handler import get_tts_handler

tts = get_tts_handler()

# Speak letter
tts.speak_letter('A')

# Speak word
tts.speak_word('HALO')

# Check apakah sedang speaking
if not tts.is_speaking:
    tts.speak_async('Siap!')
```

## Performance Notes

- ✅ Non-blocking di web mode
- ✅ Tidak mempengaruhi deteksi
- ✅ Offline - tidak perlu internet
- ✅ Low latency pada Windows
- ⚠️ Tergantung pada sistem TTS OS

---

**Happy Gesture Recognizing dengan Audio! 🎉**
