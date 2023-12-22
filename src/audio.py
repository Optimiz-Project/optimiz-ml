import librosa
import numpy as np
from scipy.signal import medfilt

def verify_audio_header(content_type):
    if content_type is not None:
        return "audio/" in content_type or "multipart/form-data" in content_type
    else:
        return False

# Fungsi untuk ekstraksi fitur MFCC
def extract_mfcc(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Fungsi untuk normalisasi fitur-fitur MFCC
def normalize_mfcc(mfcc_features):
    mean = np.mean(mfcc_features, axis=1, keepdims=True)
    std = np.std(mfcc_features, axis=1, keepdims=True)
    normalized_mfcc = (mfcc_features - mean) / (std + 1e-8)
    return normalized_mfcc

def process_audio(audio_req):
    audio, sr = librosa.load(audio_req, sr=None)
    audio = np.asarray(audio)
    mfcc = extract_mfcc(audio, sr=sr)
    normalized_mfcc = normalize_mfcc(mfcc)
    return normalized_mfcc

# # Membaca file audio
# new_audio, new_sr = librosa.load(new_audio_file, sr=None)

# # Memastikan bahwa audio dalam bentuk numpy.ndarray
# new_audio = np.asarray(new_audio)

# # Ekstraksi fitur MFCC
# new_mfcc = extract_mfcc(new_audio, sr=new_sr)

# # Normalisasi fitur-fitur MFCC
# normalized_new_mfcc = normalize_mfcc(new_mfcc)

# # Contoh: Membuat prediksi pada satu contoh audio baru
# new_audio_file = '/content/sample_data/audio/LJ001-0002.wav'  # Ganti dengan path file audio baru Anda
# new_mfcc = extract_mfcc(new_audio_file, sr=22050)  # Ganti dengan nilai sampling_rate yang sesuai
# normalized_new_mfcc = normalize_mfcc(new_mfcc)

# # Reshape data sesuai dengan bentuk input model
# input_data = normalized_new_mfcc.reshape((1, n_mfcc, target_length))
