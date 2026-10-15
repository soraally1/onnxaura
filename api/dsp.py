import os
import io
import librosa
import numpy as np
from scipy.signal import butter, lfilter

TARGET_SR = 16000

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

import tempfile

def load_and_resample(audio_bytes: bytes) -> np.ndarray:
    """
    Loads audio from bytes and resamples to 16 kHz Mono.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='/tmp') as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        temp_path = temp_audio.name
        
    try:
        y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return y

def apply_bandpass_filter(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Applies Butterworth Bandpass Filter (100 Hz - 2000 Hz).
    """
    return butter_bandpass_filter(y, 100.0, 2000.0, sr, order=5)

def extract_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extracts Mel-Spectrogram (n_mels=128, n_fft=2048, hop_length=512).
    """
    S = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=128, 
        n_fft=2048, 
        hop_length=512
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Full DSP pipeline for RespiSound AI.
    """
    y = load_and_resample(audio_bytes)
    y_filtered = apply_bandpass_filter(y)
    mel_spec = extract_mel_spectrogram(y_filtered)
    return mel_spec
