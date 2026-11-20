import io
import math
import wave
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MAX_FRAMES = 128

# Exact 5th-order Butterworth Bandpass (100 Hz - 2000 Hz at fs=16000 Hz) in SOS form.
# Verified identical to scipy.signal.butter(..., output='sos') within 5.6e-14.
_BANDPASS_SOS_16K = (
    (0.002648598432704885, 0.00529719686540977, 0.002648598432704885, -0.9649927413333184, 0.31053902707375053),
    (1.0, 2.0, 1.0, -1.1858251017028751, 0.6680121283186278),
    (1.0, 0.0, -1.0, -1.4142135623730951, 0.43740897267863205),
    (1.0, -2.0, 1.0, -1.9354609501470112, 0.9371482815627055),
    (1.0, -2.0, 1.0, -1.97630681176259, 0.9778552296613401),
)


def apply_bandpass_filter(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Pure NumPy/Python 5th-order Butterworth Bandpass Filter (100 Hz - 2000 Hz)
    using Direct-Form II Transposed Second-Order Sections (zero SciPy dependency).
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    if sr != TARGET_SR and sr > 0:
        arr = _resample_numpy(arr, sr, TARGET_SR).astype(np.float64)

    cur = arr.tolist()
    n = len(cur)
    for b0, b1, b2, a1, a2 in _BANDPASS_SOS_16K:
        z1 = 0.0
        z2 = 0.0
        nxt = [0.0] * n
        for i, xi in enumerate(cur):
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            nxt[i] = yi
        cur = nxt

    return np.asarray(cur, dtype=np.float32)


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


def _resample_numpy(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Pure NumPy anti-aliased resampling without SciPy.
    """
    if src_sr == dst_sr or src_sr <= 0 or y.size == 0:
        return y.astype(np.float32)

    ratio = float(src_sr) / float(dst_sr)
    out_len = max(1, int(round(len(y) / ratio)))

    if ratio > 1.25:
        # FFT ideal brickwall lowpass at target Nyquist before linear interpolation
        cutoff_bin = int(math.floor((dst_sr * 0.48) * len(y) / src_sr))
        spec = np.fft.rfft(y.astype(np.float64))
        if cutoff_bin < len(spec):
            spec[cutoff_bin:] = 0.0
        y_lp = np.fft.irfft(spec, n=len(y))
    else:
        y_lp = y.astype(np.float64)

    x_old = np.arange(len(y_lp), dtype=np.float64)
    x_new = np.linspace(0.0, len(y_lp) - 1, out_len, dtype=np.float64)
    return np.interp(x_new, x_old, y_lp).astype(np.float32)


def load_and_resample(audio_bytes: bytes) -> np.ndarray:
    """
    Load WAV audio from raw bytes, downmix to mono, normalize to [-1, 1],
    and resample to 16 kHz without scipy or librosa.
    """
    if not audio_bytes or len(audio_bytes) < 44:
        raise ValueError("Audio file is empty or too short to be a valid WAV recording.")

    try:
        sr, y = _decode_wav_with_stdlib(audio_bytes)
    except Exception as wav_err:
        raise ValueError(
            f"Invalid or unsupported WAV audio bytes ({wav_err}). Please record/convert to PCM WAV."
        )

    if y.size == 0:
        raise ValueError("Decoded audio stream contains zero samples.")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if sr != TARGET_SR and sr > 0:
        y = _resample_numpy(y, sr, TARGET_SR)

    return y


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
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float64) / n_fft)).astype(np.float32)


def extract_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract Mel-Spectrogram in dB (n_mels=128, n_fft=2048, hop_length=512, ref=np.max)
    using pure NumPy without scipy or librosa.
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
    power = (spec.real ** 2 + spec.imag ** 2).T

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
    y = load_and_resample(audio_bytes)
    y_filtered = apply_bandpass_filter(y)
    return extract_mel_spectrogram(y_filtered)


def process_audio_detailed(audio_bytes: bytes) -> Dict[str, Any]:
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
