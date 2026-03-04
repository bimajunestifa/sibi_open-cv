from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import mediapipe as mp
import time
from collections import Counter
import threading

app = Flask(__name__)

# ── Load model ───────────────────────────────────────────────
try:
    with open("model_bisindo_nn.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Model SIBI berhasil dimuat!")
except Exception as e:
    print(f"Error load model: {e}")
    model = le = scaler = None

# ── MediaPipe ────────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ── Normalisasi (sama dengan kumpul_data.py) ─────────────────
def normalisasi_landmark(landmarks):
    data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    data -= data[0].copy()          # geser ke wrist
    scale = np.max(np.abs(data))
    if scale > 0:
        data /= scale               # skala seragam
    return data.flatten().tolist()

# ── State ────────────────────────────────────────────────────
state = {
    "kata": "",
    "kalimat_list": [],
    "huruf": "-",
    "confidence": 0,
    "tangan_terdeteksi": False,
    "riwayat": [],
    "prediksi_window": [],
    "waktu_terakhir": time.time(),
    "huruf_terakhir": "",
    "total_deteksi": 0,
}
DELAY  = 2.0
WINDOW = 10          # jumlah frame untuk voting
lock   = threading.Lock()

# ── Frame buffer (agar video smooth) ─────────────────────────
frame_buffer = {"frame": None, "lock": threading.Lock()}

def camera_thread():
    """Thread khusus baca kamera — tidak blocking render"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # buffer minimal → frame segar
    cap.set(cv2.CAP_PROP_FPS, 30)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            tangan_ada = bool(result.hand_landmarks)

            if tangan_ada and model:
                landmarks = result.hand_landmarks[0]
                h, w, _   = frame.shape
                titik = []

                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    titik.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 150), -1)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 150), 1)

                for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                             (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                    cv2.line(frame, titik[a], titik[b], (0, 180, 255), 2)

                data_point = normalisasi_landmark(landmarks)

                if len(data_point) == 63:
                    data_norm    = scaler.transform([data_point])
                    prediksi_idx = model.predict(data_norm)[0]
                    prediksi     = le.inverse_transform([prediksi_idx])[0]
                    confidence   = float(np.max(model.predict_proba(data_norm)) * 100)

                    with lock:
                        state["prediksi_window"].append(prediksi)
                        if len(state["prediksi_window"]) > WINDOW:
                            state["prediksi_window"].pop(0)
                        prediksi_stabil = Counter(state["prediksi_window"]).most_common(1)[0][0]

                        sekarang = time.time()
                        if (prediksi_stabil == state["huruf_terakhir"] and
                                confidence > 70 and
                                sekarang - state["waktu_terakhir"] > DELAY):
                            state["kata"] += prediksi_stabil
                            state["total_deteksi"] += 1
                            state["riwayat"].insert(0, {
                                "huruf": prediksi_stabil,
                                "confidence": round(confidence, 1),
                                "waktu": time.strftime("%H:%M:%S")
                            })
                            if len(state["riwayat"]) > 30:
                                state["riwayat"].pop()
                            state["waktu_terakhir"] = sekarang

                        state["huruf_terakhir"]    = prediksi_stabil
                        state["huruf"]             = prediksi_stabil
                        state["confidence"]        = round(confidence, 1)
                        state["tangan_terdeteksi"] = True

            else:
                # ── Tangan tidak ada → reset semua ──────────
                with lock:
                    state["huruf"]             = "-"
                    state["confidence"]        = 0
                    state["tangan_terdeteksi"] = False
                    state["prediksi_window"]   = []
                    state["huruf_terakhir"]    = ""

            # Simpan frame ke buffer
            with frame_buffer["lock"]:
                frame_buffer["frame"] = frame.copy()

    cap.release()


# Jalankan thread kamera saat server start
cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()


def generate_frames():
    """Stream frame dari buffer ke browser"""
    while True:
        with frame_buffer["lock"]:
            frame = frame_buffer["frame"]

        if frame is None:
            time.sleep(0.01)
            continue

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(1/30)   # 30 fps max


# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with lock:
        sekarang = time.time()
        sisa     = max(0, DELAY - (sekarang - state["waktu_terakhir"]))
        # Timer hanya jalan kalau tangan terdeteksi
        if not state["tangan_terdeteksi"]:
            sisa = DELAY
        return jsonify({
            "huruf":             state["huruf"],
            "confidence":        state["confidence"],
            "tangan_terdeteksi": state["tangan_terdeteksi"],
            "kata":              state["kata"],
            "riwayat":           state["riwayat"][:15],
            "timer_progress":    round((1 - sisa / DELAY) * 100),
            "timer_sisa":        round(sisa, 1),
            "total_deteksi":     state["total_deteksi"],
            "kalimat_list":      state["kalimat_list"],
        })

@app.route('/hapus_kata', methods=['POST'])
def hapus_kata():
    with lock:
        state["kata"] = ""
        state["prediksi_window"] = []
    return jsonify({"ok": True})

@app.route('/hapus_huruf', methods=['POST'])
def hapus_huruf():
    with lock:
        if state["kata"]:
            state["kata"] = state["kata"][:-1]
    return jsonify({"ok": True})

@app.route('/tambah_spasi', methods=['POST'])
def tambah_spasi():
    with lock:
        state["kata"] += " "
    return jsonify({"ok": True})

@app.route('/simpan_kalimat', methods=['POST'])
def simpan_kalimat():
    with lock:
        if state["kata"].strip():
            state["kalimat_list"].insert(0, {
                "teks": state["kata"].strip(),
                "waktu": time.strftime("%H:%M:%S")
            })
            if len(state["kalimat_list"]) > 10:
                state["kalimat_list"].pop()
            state["kata"] = ""
            state["prediksi_window"] = []
    return jsonify({"ok": True})

@app.route('/hapus_riwayat', methods=['POST'])
def hapus_riwayat():
    with lock:
        state["riwayat"] = []
    return jsonify({"ok": True})

if __name__ == '__main__':
    print("="*50)
    print("  SIBI - Sistem Isyarat Bahasa Indonesia")
    print("  Buka browser: http://localhost:5000")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)