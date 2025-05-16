# Import modules
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Constants
MAX_PAD_LEN = 862
D_NAMES = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
AUDIO_PATH = "archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
MODEL_PATH = "resp_model_300.h5"

# Feature Extraction Function
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = MAX_PAD_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return None

# Collect .wav file paths
file_names = [f for f in listdir(AUDIO_PATH) if isfile(join(AUDIO_PATH, f)) and f.endswith('.wav')]
file_paths = [join(AUDIO_PATH, f) for f in file_names]

# Feature extraction loop
features = []
valid_files = []

for file_name in file_paths[71:]:
    feat = extract_features(file_name)
    if feat is not None:
        features.append(feat)
        valid_files.append(file_name)
    else:
        print(f"Skipping invalid file: {file_name}")
    
    if len(features) == 21:
        break

print(f"\n✅ Finished feature extraction from {len(features)} files.")

# Convert to NumPy array
features = np.array(features, dtype=np.float32)
print("Features shape:", features.shape)  # Expected: (21, 40, 862)

# Load the trained model
print("\n📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Predict and display results
for i in range(len(features)):
    sample = np.expand_dims(features[i], axis=0)  # Shape (1, 40, 862)
    result = model.predict(sample)

    print(f"\n🔍 Prediction for file: {valid_files[i].split('/')[-1]}")
    dp = list(zip(D_NAMES, list(*result)))

    for disease, prob in dp:
        print(f"{disease} : {prob*100:.2f}%")

    predicted = max(dp, key=lambda x: x[1])
    print(f"➡️ Predicted Disease: {predicted[0]} ({predicted[1]*100:.2f}%)")
