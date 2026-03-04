import cv2
import numpy as np
import os
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
INPUT_DIR = "indonesian-sign-language-bisindo/Data/datatest"
OUTPUT_DIR = "data_gesture"

os.makedirs(OUTPUT_DIR, exist_ok=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

total_berhasil = 0
total_gagal = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for huruf in sorted(os.listdir(INPUT_DIR)):
        folder_input = os.path.join(INPUT_DIR, huruf)
        if not os.path.isdir(folder_input):
            continue

        folder_output = os.path.join(OUTPUT_DIR, huruf.upper())
        os.makedirs(folder_output, exist_ok=True)

        gambar_list = [f for f in os.listdir(folder_input)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Memproses huruf {huruf}: {len(gambar_list)} gambar...")
        counter = 0

        for nama_file in gambar_list:
            path_gambar = os.path.join(folder_input, nama_file)
            frame = cv2.imread(path_gambar)
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                data_point = []
                for lm in landmarks:
                    data_point.extend([lm.x, lm.y, lm.z])

                if len(data_point) == 63:
                    np.save(os.path.join(folder_output, f"{counter}.npy"), data_point)
                    counter += 1
                    total_berhasil += 1
            else:
                total_gagal += 1

        print(f"  Huruf {huruf}: {counter} sampel berhasil dikonversi")

print(f"\nSelesai! Total berhasil: {total_berhasil}, Gagal: {total_gagal}")
print("Sekarang jalankan: python latih_model.py")