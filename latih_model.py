import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data_gesture"
HURUF = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("Memuat data...")
data = []
labels = []

for huruf in HURUF:
    folder = os.path.join(DATA_DIR, huruf)
    if not os.path.exists(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if len(files) == 0:
        continue
    for file in files:
        try:
            sample = np.load(os.path.join(folder, file))
            if len(sample) == 63:
                data.append(sample)
                labels.append(huruf)
        except:
            continue
    print(f"Huruf {huruf}: {len(files)} sampel")

print(f"\nTotal data: {len(data)} sampel")
print(f"Total huruf: {len(set(labels))} huruf")

if len(data) == 0:
    print("Tidak ada data!")
    exit()

X = np.array(data)
y = np.array(labels)

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Kelas: {le.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalisasi
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Augmentasi data
print("\nMelakukan augmentasi data...")
X_aug = [X_train_norm]
y_aug = [y_train]
for _ in range(4):
    noise = np.random.normal(0, 0.01, X_train_norm.shape)
    X_aug.append(X_train_norm + noise)
    y_aug.append(y_train)
X_train_aug = np.vstack(X_aug)
y_train_aug = np.concatenate(y_aug)
print(f"Data training setelah augmentasi: {len(X_train_aug)} sampel")

# Model 1: Neural Network (MLP)
print("\nMelatih Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)
mlp.fit(X_train_aug, y_train_aug)

# Model 2: Random Forest
print("\nMelatih Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_aug, y_train_aug)

# Voting Classifier (gabungan kedua model)
print("\nMenggabungkan model (Voting)...")
voting = VotingClassifier(
    estimators=[('mlp', mlp), ('rf', rf)],
    voting='soft'
)
voting.fit(X_train_aug, y_train_aug)

# Evaluasi
y_pred = voting.predict(X_test_norm)
akurasi = accuracy_score(y_test, y_pred)
print(f"\nAkurasi model gabungan: {akurasi * 100:.2f}%")
print("\nLaporan per huruf:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Simpan semua
with open("model_bisindo_nn.pkl", "wb") as f:
    pickle.dump(voting, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel disimpan!")
print("Jalankan: python deteksi.py")