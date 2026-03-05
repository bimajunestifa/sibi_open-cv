import cv2
import numpy as np
import os
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
DATA_DIR   = "data_gesture"

# ── Konfigurasi ───────────────────────────────────────────────
JUMLAH_SAMPEL = 300          # naik dari 200 → 300
HURUF         = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Huruf yang bentuknya mirip — diberi perhatian ekstra
HURUF_MIRIP = {
    'A': ['S'],   'B': ['D'],   'C': ['G', 'O'],
    'D': ['B'],   'G': ['C'],   'M': ['N'],
    'N': ['M'],   'O': ['C'],   'P': ['Q'],
    'Q': ['P'],   'R': ['U'],   'S': ['A'],
    'T': ['I'],   'U': ['R', 'V'], 'V': ['U'],
}

os.makedirs(DATA_DIR, exist_ok=True)
for huruf in HURUF:
    os.makedirs(os.path.join(DATA_DIR, huruf), exist_ok=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


def normalisasi_landmark(landmarks):
    data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    data -= data[0].copy()
    scale = np.max(np.abs(data))
    if scale > 0:
        data /= scale
    return data.flatten().tolist()


def augmentasi_satu_sampel(data_point: list) -> list:
    """
    Hasilkan variasi kecil dari 1 sampel agar model
    toleran terhadap posisi tangan yang tidak persis sama.
    """
    arr = np.array(data_point)
    variasi = []

    # 1. Noise ringan (getaran tangan)
    for sigma in [0.008, 0.015]:
        variasi.append((arr + np.random.normal(0, sigma, arr.shape)).tolist())

    # 2. Skala sedikit lebih besar / kecil (jarak ke kamera)
    for s in [0.95, 1.05]:
        variasi.append((arr * s).tolist())

    # 3. Geser sedikit (posisi tangan tidak di tengah)
    for dx in [-0.03, 0.03]:
        shifted = arr.copy()
        shifted[0::3] += dx          # geser sumbu-x semua titik
        variasi.append(shifted.tolist())

    return variasi                   # 6 variasi per sampel asli


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Webcam siap! Rekam 300 sampel per huruf.")
print("Tips: variasikan sudut & jarak tangan agar model lebih toleran.\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for huruf in HURUF:
        folder   = os.path.join(DATA_DIR, huruf)
        existing = len([f for f in os.listdir(folder) if f.endswith(".npy")])

        if existing >= JUMLAH_SAMPEL:
            print(f"Huruf {huruf} sudah lengkap ({existing} sampel), skip.")
            continue

        mirip_dengan = HURUF_MIRIP.get(huruf, [])
        pesan_mirip  = (f"  ⚠  Mirip dengan: {', '.join(mirip_dengan)} — pastikan gestur jelas!"
                        if mirip_dengan else "")

        print(f"\n{'='*50}")
        print(f"  Huruf: {huruf}  ({existing}/{JUMLAH_SAMPEL} sudah ada)")
        if pesan_mirip:
            print(pesan_mirip)
        print(f"  Tekan SPASI untuk mulai, Q untuk skip")
        print(f"{'='*50}")

        # ── Layar tunggu ──────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)

            # Huruf besar di tengah
            cv2.putText(frame, huruf, (w//2 - 60, h//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 150), 8)

            # Info mirip
            if mirip_dengan:
                cv2.putText(frame,
                            f"Mirip: {', '.join(mirip_dengan)} — buat gestur sejelas mungkin!",
                            (20, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            # Panduan variasi
            panduan = [
                "Variasikan posisi tangan:",
                "  1. Lurus ke kamera",
                "  2. Sedikit miring kiri",
                "  3. Sedikit miring kanan",
            ]
            for i, teks in enumerate(panduan):
                cv2.putText(frame, teks, (20, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (180, 180, 180), 1)

            cv2.putText(frame, "SPASI = mulai  |  Q = skip",
                        (w//2 - 160, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Kumpul Data SIBI", frame)
            key = cv2.waitKey(1)
            if key == ord(' ') or key == ord('q'):
                break

        if key == ord('q'):
            continue

        # ── Rekam sampel ──────────────────────────────────────
        counter    = existing
        aug_counter = 0       # hitung augmentasi yang disimpan

        while counter < JUMLAH_SAMPEL:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks  = result.hand_landmarks[0]
                data_point = normalisasi_landmark(landmarks)

                if len(data_point) == 63:
                    # Simpan sampel asli
                    np.save(os.path.join(DATA_DIR, huruf, f"{counter}.npy"),
                            data_point)
                    counter += 1

                    # Simpan variasi augmentasi (tidak dihitung ke counter utama
                    # agar tidak menggantikan sampel asli, tapi disimpan sebagai
                    # sampel tambahan dengan nama aug_*)
                    variasi = augmentasi_satu_sampel(data_point)
                    for v in variasi:
                        aug_path = os.path.join(
                            DATA_DIR, huruf, f"aug_{aug_counter}.npy")
                        np.save(aug_path, v)
                        aug_counter += 1

                # Gambar skeleton
                titik = []
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    titik.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 150), -1)
                for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                             (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                    cv2.line(frame, titik[a], titik[b], (0, 180, 255), 2)

            # UI progress
            progress = int((counter / JUMLAH_SAMPEL) * (w - 20))
            cv2.rectangle(frame, (10, h-30), (w-10, h-10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h-30), (10+progress, h-10),
                          (0, 255, 150), -1)
            cv2.putText(frame,
                        f"{huruf}: {counter}/{JUMLAH_SAMPEL}  (+{aug_counter} aug)",
                        (10, h-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Peringatan mirip di layar rekam
            if mirip_dengan:
                cv2.putText(frame,
                            f"Mirip: {', '.join(mirip_dengan)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Kumpul Data SIBI", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        total_file = len([f for f in os.listdir(folder) if f.endswith(".npy")])
        print(f"Selesai huruf {huruf}: {counter} asli + {aug_counter} aug = {total_file} total")

cap.release()
cv2.destroyAllWindows()
print("\nSemua data selesai direkam!")
print("Jalankan: python latih_model.py")