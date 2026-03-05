import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data_gesture"
HURUF    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Pasangan huruf yang bentuknya mirip — dipakai untuk analisis confusion
HURUF_MIRIP = {
    'A': ['S'],   'B': ['D'],   'C': ['G', 'O'],
    'D': ['B'],   'G': ['C'],   'M': ['N'],
    'N': ['M'],   'O': ['C'],   'P': ['Q'],
    'Q': ['P'],   'R': ['U'],   'S': ['A'],
    'T': ['I'],   'U': ['R', 'V'], 'V': ['U'],
}

# ── 1. MUAT DATA ──────────────────────────────────────────────
print("="*55)
print("  SIBI — Pelatihan Model")
print("="*55)
print("\n[1/5] Memuat data...")

data, labels = [], []
for huruf in HURUF:
    folder = os.path.join(DATA_DIR, huruf)
    if not os.path.exists(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        continue
    ok = 0
    for file in files:
        try:
            sample = np.load(os.path.join(folder, file))
            if len(sample) == 63:
                data.append(sample)
                labels.append(huruf)
                ok += 1
        except:
            continue
    print(f"  {huruf}: {ok} sampel")

print(f"\n  Total: {len(data)} sampel, {len(set(labels))} huruf")

if not data:
    print("Tidak ada data! Jalankan kumpul_data.py dulu.")
    exit()

X = np.array(data)
y = np.array(labels)

# ── 2. ENCODE & SPLIT ────────────────────────────────────────
print("\n[2/5] Menyiapkan data...")
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Kelas: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. AUGMENTASI AGRESIF ────────────────────────────────────
print("\n[3/5] Augmentasi data training...")

def augmentasi(X, y, n_kali=6):
    """
    Augmentasi dengan noise + skala + geser.
    Lebih agresif untuk huruf-huruf yang mirip.
    """
    aug_X, aug_y = [X], [y]

    for _ in range(n_kali):
        # Noise ringan
        aug_X.append(X + np.random.normal(0, 0.012, X.shape))
        aug_y.append(y)

    # Skala kecil/besar (variasi jarak tangan)
    for s in [0.93, 0.97, 1.03, 1.07]:
        aug_X.append(X * s)
        aug_y.append(y)

    # Geser sedikit (posisi tangan tidak di tengah)
    for dx in [-0.04, -0.02, 0.02, 0.04]:
        shifted = X.copy()
        shifted[:, 0::3] += dx
        aug_X.append(shifted)
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)

X_aug, y_aug = augmentasi(X_train_sc, y_train)
print(f"  Sebelum: {len(X_train_sc)} → Sesudah: {len(X_aug)} sampel")

# ── 4. LATIH MODEL ───────────────────────────────────────────
print("\n[4/5] Melatih model...")

# MLP — lebih dalam & dropout-like via noise augmentasi
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),  # lebih dalam
    activation='relu',
    solver='adam',
    alpha=0.001,             # regularisasi lebih kuat → toleran variasi
    learning_rate='adaptive',
    max_iter=600,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False            # matikan verbose agar tidak spam
)

# Random Forest — lebih banyak pohon, kedalaman terbatas
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,            # tidak terlalu dalam → lebih general
    min_samples_leaf=2,      # tidak overfit ke 1 sampel
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # ← huruf yang datanya sedikit tetap diperhatikan
)

print("  Melatih MLP...")
mlp.fit(X_aug, y_aug)
print(f"  MLP selesai (iterasi: {mlp.n_iter_})")

print("  Melatih Random Forest...")
rf.fit(X_aug, y_aug)
print("  RF selesai")

print("  Menggabungkan model (Soft Voting)...")
voting = VotingClassifier(
    estimators=[('mlp', mlp), ('rf', rf)],
    voting='soft',
    weights=[2, 1]           # MLP diberi bobot lebih besar
)
voting.fit(X_aug, y_aug)
print("  Voting selesai")

# ── 5. EVALUASI ───────────────────────────────────────────────
print("\n[5/5] Evaluasi model...")

y_pred = voting.predict(X_test_sc)
akurasi = accuracy_score(y_test, y_pred)
print(f"\n  Akurasi keseluruhan: {akurasi * 100:.2f}%")

print("\n  Laporan per huruf:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Analisis khusus huruf mirip
print("\n  Analisis huruf mirip:")
cm = confusion_matrix(y_test, y_pred)
ada_masalah = False
for huruf, mirip_list in HURUF_MIRIP.items():
    if huruf not in le.classes_:
        continue
    idx_a = list(le.classes_).index(huruf)
    for m in mirip_list:
        if m not in le.classes_:
            continue
        idx_b = list(le.classes_).index(m)
        salah_ab = cm[idx_a][idx_b]   # A dikira B
        salah_ba = cm[idx_b][idx_a]   # B dikira A
        if salah_ab > 0 or salah_ba > 0:
            ada_masalah = True
            print(f"  ⚠  {huruf}→{m}: {salah_ab}x  |  {m}→{huruf}: {salah_ba}x")

if not ada_masalah:
    print("  Tidak ada kebingungan antar huruf mirip!")

# ── SIMPAN ────────────────────────────────────────────────────
with open("model_bisindo_nn.pkl", "wb") as f:
    pickle.dump(voting, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n" + "="*55)
print("  Model disimpan!")
print(f"  Akurasi akhir: {akurasi * 100:.2f}%")
print("="*55)
print("\nJalankan: python deteksi.py  atau  python app.py")