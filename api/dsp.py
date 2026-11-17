import io
import math
import wave
from functools import lru_cache
from typing import Dict, Any, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, get_window, lfilter, resample_poly

TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MAX_FRAMES = 128


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data).astype(np.float32)


def _decode_wav_with_stdlib(audio_bytes: bytes) -> Tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return sr, arr.astype(np.float32)


def _normalize_wav_array(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.floating):
        arr = data.astype(np.float32)
    elif data.dtype == np.uint8:
        arr = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:
        arr = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        arr = data.astype(np.float32) / 2147483648.0
    else:
        max_abs = float(np.max(np.abs(data))) or 1.0
        arr = data.astype(np.float32) / max_abs

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_and_resample(audio_bytes: bytes) -> np.ndarray:
    """
    Load WAV audio from raw bytes, downmix to mono, normalize to [-1, 1],
    and resample to 16 kHz without librosa/numba stubs.
    """
    if not audio_bytes or len(audio_bytes) < 44:
        raise ValueError("Audio file is empty or too short to be a valid WAV recording.")

    try:
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        y = _normalize_wav_array(data)
    except Exception as wav_err:
        try:
            sr, y = _decode_wav_with_stdlib(audio_bytes)
        except Exception:
            raise ValueError(
                f"Invalid or unsupported WAV audio bytes ({wav_err}). Please record/convert to PCM WAV."
            )

    if y.size == 0:
        raise ValueError("Decoded audio stream contains zero samples.")

    if sr != TARGET_SR and sr > 0:
        divisor = math.gcd(int(sr), int(TARGET_SR))
        up = int(TARGET_SR) // divisor
        down = int(sr) // divisor
        y = resample_poly(y, up, down).astype(np.float32)

    return y


def apply_bandpass_filter(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Apply 5th-order Butterworth Bandpass Filter (100 Hz - 2000 Hz)
    for lung & cough acoustic frequencies.
    """
    return butter_bandpass_filter(y, 100.0, 2000.0, float(sr), order=5)


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = np.asarray(frequencies, dtype=np.float64)
    mels = (freqs - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if freqs.ndim:
        log_t = freqs >= min_log_hz
        mels[log_t] = min_log_mel + np.log(freqs[log_t] / min_log_hz) / logstep
    elif freqs >= min_log_hz:
        mels = min_log_mel + np.log(freqs / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    m = np.asarray(mels, dtype=np.float64)
    freqs = f_min + f_sp * m
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if m.ndim:
        log_t = m >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (m[log_t] - min_log_mel))
    elif m >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (m - min_log_mel))
    return freqs


@lru_cache(maxsize=4)
def get_slaney_mel_meta(
    sr: int = TARGET_SR, n_fft: int = N_FFT, n_mels: int = N_MELS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (mel_filterbank [n_mels, 1 + n_fft//2], mel_center_freqs_hz [n_mels]).
    Matches librosa.filters.mel(..., htk=False, norm='slaney') to ~1e-8 precision.
    """
    fmax = float(sr) / 2.0
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float64)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    min_mel = _hz_to_mel_slaney(np.asarray(0.0))
    max_mel = _hz_to_mel_slaney(np.asarray(fmax))
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = _mel_to_hz_slaney(mels)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    center_freqs = mel_f[1 : n_mels + 1].astype(np.float32)
    return weights.astype(np.float32), center_freqs


@lru_cache(maxsize=2)
def _get_hann_window(n_fft: int = N_FFT) -> np.ndarray:
    return get_window("hann", n_fft, fftbins=True).astype(np.float32)


def extract_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract Mel-Spectrogram in dB (n_mels=128, n_fft=2048, hop_length=512, ref=np.max)
    using pure NumPy/SciPy without librosa.
    """
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.size < 1:
        return np.zeros((N_MELS, 1), dtype=np.float32)

    y_pad = np.pad(y_arr, N_FFT // 2, mode="constant")
    if len(y_pad) < N_FFT:
        y_pad = np.pad(y_pad, (0, N_FFT - len(y_pad)), mode="constant")

    frames = np.lib.stride_tricks.sliding_window_view(y_pad, N_FFT)[::HOP_LENGTH]
    win = _get_hann_window(N_FFT)
    spec = np.fft.rfft(frames * win, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).T  # shape: (1 + N_FFT//2, T)

    fb, _ = get_slaney_mel_meta(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel_s = fb @ power

    amin = 1e-10
    ref_val = max(amin, float(np.max(mel_s)))
    mel_db = 10.0 * np.log10(np.maximum(amin, mel_s)) - 10.0 * np.log10(ref_val)
    mel_db = np.maximum(mel_db, float(np.max(mel_db)) - 80.0)
    return mel_db.astype(np.float32)


def fit_mel_to_model_window(mel_spec: np.ndarray, max_length: int = MAX_FRAMES) -> np.ndarray:
    if mel_spec.shape[1] > max_length:
        return mel_spec[:, :max_length]
    pad_width = max_length - mel_spec.shape[1]
    return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")


def process_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Full DSP pipeline returning raw Mel-Spectrogram for ICBHI ONNX inference.
    """
    y = load_and_resample(audio_bytes)
    y_filtered = apply_bandpass_filter(y)
    return extract_mel_spectrogram(y_filtered)


def process_audio_detailed(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Full DSP pipeline returning waveform, filtered signal, exact ONNX input window,
    and frequency metadata for clinical spectrogram visualization & explanation.
    """
    y_raw = load_and_resample(audio_bytes)
    y_filtered = apply_bandpass_filter(y_raw)
    mel_raw = extract_mel_spectrogram(y_filtered)
    mel_model = fit_mel_to_model_window(mel_raw, MAX_FRAMES)
    _, mel_center_freqs = get_slaney_mel_meta(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS)
    return {
        "y_raw": y_raw,
        "y_filtered": y_filtered,
        "mel_raw": mel_raw,
        "mel_model": mel_model,
        "mel_center_freqs": mel_center_freqs,
        "sample_rate": TARGET_SR,
        "duration_sec": round(len(y_raw) / float(TARGET_SR), 2),
        "window_sec": round((MAX_FRAMES * HOP_LENGTH) / float(TARGET_SR), 2),
    }
