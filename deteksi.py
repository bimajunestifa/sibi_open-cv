import cv2
import numpy as np
import pickle
import mediapipe as mp
import time
import pyttsx3

# ── TTS Engine untuk suara ───────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

def speak_letter(letter):
    """Bicara huruf dengan suara"""
    if letter and letter != "-":
        try:
            tts_engine.say(letter)
            tts_engine.runAndWait()
        except:
            pass

# Load model baru
with open("model_bisindo_nn.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Variabel
kata = ""
huruf_terakhir = ""
waktu_terakhir = time.time()
DELAY = 2.0
riwayat_prediksi = []
WINDOW_PREDIKSI = 10

cap = cv2.VideoCapture(0)
print("Aplikasi BISINDO siap! Tekan Q untuk keluar.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        prediksi = "-"
        confidence = 0
        prediksi_stabil = "-"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            h, w, _ = frame.shape

            # Gambar koneksi tangan
            titik = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                titik.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Gambar garis koneksi
            koneksi = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for a, b in koneksi:
                cv2.line(frame, titik[a], titik[b], (0, 200, 255), 2)

            # Prediksi
            data_point = []
            for lm in landmarks:
                data_point.extend([lm.x, lm.y, lm.z])

            if len(data_point) == 63:
                data_norm = scaler.transform([data_point])
                prediksi_idx = model.predict(data_norm)[0]
                prediksi = le.inverse_transform([prediksi_idx])[0]
                proba = model.predict_proba(data_norm)
                confidence = np.max(proba) * 100

                # Stabilkan prediksi dengan voting window
                riwayat_prediksi.append(prediksi)
                if len(riwayat_prediksi) > WINDOW_PREDIKSI:
                    riwayat_prediksi.pop(0)

                from collections import Counter
                if riwayat_prediksi:
                    prediksi_stabil = Counter(riwayat_prediksi).most_common(1)[0][0]

                # Tambahkan huruf ke kata
                sekarang = time.time()
                if (prediksi_stabil == huruf_terakhir and
                        confidence > 70 and
                        sekarang - waktu_terakhir > DELAY):
                    kata += prediksi_stabil
                    waktu_terakhir = sekarang
                    # 🔊 PUTAR SUARA KETIKA HURUF TERDETEKSI
                    speak_letter(prediksi_stabil)

                huruf_terakhir = prediksi_stabil

                # Timer visual
                elapsed = time.time() - waktu_terakhir
                sisa = max(0, DELAY - elapsed)
                progress = int((1 - sisa / DELAY) * 200)
                cv2.rectangle(frame, (10, 130), (210, 150), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 130), (10 + progress, 150), (0, 255, 100), -1)
                cv2.putText(frame, f"Tahan: {sisa:.1f}s", (215, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # UI Panel atas
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 160), (20, 20, 20), -1)

        # Huruf terdeteksi
        cv2.putText(frame, f"Huruf: {prediksi_stabil}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        # Confidence bar
        conf_w = int(confidence * 2)
        cv2.rectangle(frame, (200, 20), (400, 50), (50, 50, 50), -1)
        warna_conf = (0, 255, 0) if confidence > 70 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
        cv2.rectangle(frame, (200, 20), (200 + conf_w, 50), warna_conf, -1)
        cv2.putText(frame, f"{confidence:.1f}%", (405, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Kata terbentuk
        cv2.putText(frame, f"Kata: {kata}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # Panel bawah
        cv2.rectangle(frame, (0, frame.shape[0]-35),
                      (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        cv2.putText(frame, "Q: Keluar | SPASI: Hapus Kata | ENTER: Hapus Huruf Terakhir",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Deteksi BISINDO - Real Time", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            kata = ""
            riwayat_prediksi = []
        elif key == 13:
            if len(kata) > 0:
                kata = kata[:-1]

cap.release()
cv2.destroyAllWindows()
print(f"Sesi selesai. Kata terakhir: {kata}")