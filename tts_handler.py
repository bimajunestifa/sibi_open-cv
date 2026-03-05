"""
Text-to-Speech Handler untuk SIBI
Menggunakan gTTS (Google) untuk suara Bahasa Indonesia
Fallback ke pyttsx3 jika tidak ada koneksi internet
"""

import threading
import os
import io
import tempfile

# ── Coba import gTTS + pygame ─────────────────────────────────
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    GTTS_AVAILABLE = True
    print("✅ gTTS + pygame siap (suara Bahasa Indonesia)")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️  gTTS tidak tersedia, fallback ke pyttsx3 (English)")

# ── Fallback pyttsx3 ──────────────────────────────────────────
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# ── Cache audio agar huruf yang sama tidak fetch ulang ────────
# Key: huruf/teks, Value: path file .mp3 temp
_audio_cache: dict = {}
_cache_lock = threading.Lock()


def _buat_audio_gtts(teks: str) -> str | None:
    """Buat file audio dengan gTTS, return path file temp"""
    with _cache_lock:
        if teks in _audio_cache:
            return _audio_cache[teks]

    try:
        tts = gTTS(text=teks, lang='id', slow=False)
        tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tts.save(tmp.name)
        with _cache_lock:
            _audio_cache[teks] = tmp.name
        return tmp.name
    except Exception as e:
        print(f"gTTS error: {e}")
        return None


class TTSHandler:
    def __init__(self):
        self._pyttsx3_eng = None
        self._antrian     = []          # antrian huruf yang akan diucapkan
        self._antrian_lock = threading.Lock()
        self._worker_running = False

        if PYTTSX3_AVAILABLE and not GTTS_AVAILABLE:
            try:
                self._pyttsx3_eng = pyttsx3.init()
                self._pyttsx3_eng.setProperty('rate',   130)
                self._pyttsx3_eng.setProperty('volume', 1.0)
            except Exception as e:
                print(f"pyttsx3 init error: {e}")

        # Pre-cache A-Z di background
        if GTTS_AVAILABLE:
            threading.Thread(target=self._precache_huruf,
                             daemon=True).start()

        # Jalankan worker thread permanen
        threading.Thread(target=self._worker_loop,
                         daemon=True).start()

    def _precache_huruf(self):
        import string
        print("Pre-cache audio huruf A-Z...")
        for huruf in string.ascii_uppercase:
            if huruf not in _audio_cache:
                _buat_audio_gtts(huruf)
        print("Pre-cache selesai — semua huruf siap diucapkan")

    # ── PUBLIC API ────────────────────────────────────────────

    def speak_letter(self, letter: str):
        """Masukkan huruf ke antrian — tidak pernah blocking"""
        if not letter or letter in ("-", "?"):
            return
        letter = letter.upper().strip()
        with self._antrian_lock:
            # Kosongkan antrian lama, langsung ganti dengan huruf baru
            # sehingga selalu ucapkan huruf terbaru
            self._antrian.clear()
            self._antrian.append(letter)

    def speak_word(self, word: str):
        if word and word.strip():
            with self._antrian_lock:
                self._antrian.clear()
                self._antrian.append(word.strip())

    def speak_async(self, text: str):
        self.speak_letter(text)

    def stop(self):
        with self._antrian_lock:
            self._antrian.clear()
        try:
            if GTTS_AVAILABLE:
                pygame.mixer.music.stop()
            elif self._pyttsx3_eng:
                self._pyttsx3_eng.stop()
        except:
            pass

    # ── INTERNAL: worker loop permanen ───────────────────────

    def _worker_loop(self):
        """
        Thread ini jalan terus selama program hidup.
        Ambil huruf dari antrian → ucapkan → ambil lagi.
        Tidak ada lock acquire/release yang bisa macet.
        """
        import time as _time
        while True:
            teks = None
            with self._antrian_lock:
                if self._antrian:
                    teks = self._antrian.pop(0)

            if teks:
                try:
                    if GTTS_AVAILABLE:
                        self._play_gtts(teks)
                    elif self._pyttsx3_eng:
                        self._play_pyttsx3(teks)
                except Exception as e:
                    print(f"TTS worker error: {e}")
            else:
                _time.sleep(0.05)   # idle — hemat CPU

    def _play_gtts(self, teks: str):
        import time as _time
        try:
            path = _buat_audio_gtts(teks)
            if not path or not os.path.exists(path):
                return
            # Gunakan Sound object agar tidak konflik dengan music channel
            sound = pygame.mixer.Sound(path)
            sound.play()
            # Tunggu selesai
            while pygame.mixer.get_busy():
                _time.sleep(0.03)
        except Exception as e:
            print(f"Play gTTS error: {e}")

    def _play_pyttsx3(self, teks: str):
        try:
            self._pyttsx3_eng.say(teks)
            self._pyttsx3_eng.runAndWait()
        except Exception as e:
            print(f"pyttsx3 error: {e}")


# ── Singleton ─────────────────────────────────────────────────
_tts_instance = None

def get_tts_handler() -> TTSHandler:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSHandler()
    return _tts_instance