import cv2
import numpy as np
import os
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
DATA_DIR = "data_gesture"
JUMLAH_SAMPEL = 200
HURUF = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

os.makedirs(DATA_DIR, exist_ok=True)
for huruf in HURUF:
    os.makedirs(os.path.join(DATA_DIR, huruf), exist_ok=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


def normalisasi_landmark(landmarks):
    """Normalisasi agar posisi & ukuran tangan tidak pengaruhi prediksi"""
    data = []
    for lm in landmarks:
        data.append([lm.x, lm.y, lm.z])
    data = np.array(data)

    # Geser ke wrist sebagai pusat
    wrist = data[0].copy()
    data -= wrist

    # Skala seragam
    scale = np.max(np.abs(data))
    if scale > 0:
        data /= scale

    return data.flatten().tolist()


cap = cv2.VideoCapture(0)
print("Webcam siap! Rekam 200 sampel per huruf.")

with HandLandmarker.create_from_options(options) as landmarker:
    for huruf in HURUF:
        folder = os.path.join(DATA_DIR, huruf)
        existing = len([f for f in os.listdir(folder) if f.endswith(".npy")])

        if existing >= JUMLAH_SAMPEL:
            print(f"Huruf {huruf} sudah lengkap ({existing} sampel), skip.")
            continue

        print(f"\nSiapkan gesture untuk huruf: {huruf} (sudah ada {existing})")
        print("Tekan SPASI untuk mulai, Q untuk skip")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.putText(frame, f"Huruf: {huruf}", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 150), 3)
            cv2.putText(frame, f"Sampel: {existing}/{JUMLAH_SAMPEL}  |  SPASI=mulai  Q=skip",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.imshow("Kumpul Data SIBI", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            elif key == ord('q'):
                break

        if key == ord('q'):
            continue

        counter = existing
        while counter < JUMLAH_SAMPEL:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                titik = []
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    titik.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 150), -1)
                    cv2.circle(frame, (cx, cy), 9, (0, 255, 150), 1)

                for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                             (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                    cv2.line(frame, titik[a], titik[b], (0, 180, 255), 2)

                # Simpan dengan normalisasi
                data_point = normalisasi_landmark(landmarks)

                if len(data_point) == 63:
                    np.save(os.path.join(DATA_DIR, huruf, f"{counter}.npy"), data_point)
                    counter += 1

            # Progress bar
            progress = int((counter / JUMLAH_SAMPEL) * (w - 20))
            cv2.rectangle(frame, (10, h-30), (w-10, h-10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h-30), (10+progress, h-10), (0, 255, 150), -1)
            cv2.putText(frame, f"{huruf}: {counter}/{JUMLAH_SAMPEL}", (10, h-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Kumpul Data SIBI", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        print(f"Selesai huruf {huruf}: {counter} sampel")

cap.release()
cv2.destroyAllWindows()
print("Semua data selesai direkam!")