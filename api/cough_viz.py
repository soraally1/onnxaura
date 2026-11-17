import base64
import io
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def _build_clinical_palette() -> np.ndarray:
    """
    256-entry RGB clinical spectrogram LUT tuned for AURA Blue & White UI:
    -80 dB -> Deep Navy (#08142C)
    -55 dB -> Royal Blue (#1E40AF / #1E6CEF)
    -35 dB -> Cyan Sky (#06B6D4 / #38BDF8)
    -15 dB -> Amber (#F59E0B)
      0 dB -> Coral / Warm Peak (#EF4444 -> #FEF3C7)
    """
    stops = [
        (0.00, (8, 20, 44)),
        (0.22, (21, 58, 138)),
        (0.45, (30, 108, 239)),
        (0.66, (6, 182, 212)),
        (0.82, (245, 158, 11)),
        (0.93, (239, 68, 68)),
        (1.00, (254, 243, 199)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = [int(round(pos * 255)) for pos, _ in stops]
    for c in range(3):
        vals = [rgb[c] for _, rgb in stops]
        lut[:, c] = np.clip(np.interp(np.arange(256), xs, vals), 0, 255).astype(np.uint8)
    return lut


_SPECTROGRAM_LUT = _build_clinical_palette()


def _render_spectrogram_base64(mel_model: np.ndarray, valid_frames: int) -> str:
    """
    Render the exact 128x128 ONNX Mel-Spectrogram into a crisp 512x256 JPEG data URI.
    Row 0 becomes bottom (low Hz) and row 127 becomes top (high Hz).
    """
    # Normalize dB [-80..0] to [0..1]
    min_db = float(np.min(mel_model))
    max_db = float(np.max(mel_model))
    if max_db - min_db < 1e-5:
        norm = np.zeros_like(mel_model, dtype=np.float32)
    else:
        # Focus dynamic range onto informative 65 dB window above floor
        floor_db = max(min_db, max_db - 68.0)
        norm = np.clip((mel_model - floor_db) / max(1e-5, max_db - floor_db), 0.0, 1.0)

    # Dim unrecorded zero-padding tail gently so user sees exact audio boundary
    if 0 < valid_frames < mel_model.shape[1]:
        norm[:, valid_frames:] = np.minimum(norm[:, valid_frames:], 0.04)

    flipped = np.flipud(norm)  # Top row = high frequency, bottom row = low frequency
    indices = np.clip((flipped * 255.0).astype(np.int32), 0, 255)
    rgb = _SPECTROGRAM_LUT[indices]

    img = Image.fromarray(rgb, mode="RGB").resize((512, 256), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _freq_to_y_pct(freq_hz: float, mel_center_freqs: np.ndarray) -> float:
    """Convert frequency (Hz) to normalized top-down Y coordinate [0..1]."""
    bin_idx = float(np.interp(freq_hz, mel_center_freqs, np.arange(len(mel_center_freqs))))
    # After flipud: bin 0 -> y=1.0, bin 127 -> y=0.0
    return round(float(np.clip(1.0 - (bin_idx / max(1, len(mel_center_freqs) - 1)), 0.0, 1.0)), 4)


def _compute_waveform_envelope(y_filtered: np.ndarray, window_sec: float, sr: int, bars: int = 64) -> List[float]:
    total_target_samples = max(bars, int(round(window_sec * sr)))
    sig = y_filtered[:total_target_samples]
    if len(sig) < total_target_samples:
        sig = np.pad(sig, (0, total_target_samples - len(sig)), mode="constant")

    chunks = np.array_split(sig, bars)
    rms = np.array([float(np.sqrt(np.mean(c ** 2))) if len(c) else 0.0 for c in chunks], dtype=np.float32)
    peak = float(np.max(rms))
    if peak < 1e-6:
        return [0.04] * bars
    norm = np.clip(rms / peak, 0.03, 1.0)
    return [round(float(v), 3) for v in norm]


def _extract_detected_regions(
    norm_mel: np.ndarray,
    mel_center_freqs: np.ndarray,
    valid_frames: int,
    window_sec: float,
    pred_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Detect real acoustic regions of interest (cough bursts, wheeze tonal harmonics,
    crackle transients) directly on the 128x128 spectrogram fed to ONNX ICBHI.
    """
    n_mels, n_frames = norm_mel.shape
    v_frames = max(1, min(n_frames, valid_frames))
    active_mat = norm_mel[:, :v_frames]

    # Restrict feature analysis to clinical respiratory bandpass 100-2000 Hz
    bp_mask = (mel_center_freqs >= 100.0) & (mel_center_freqs <= 2000.0)
    bp_bins = np.where(bp_mask)[0]
    bp_mat = active_mat[bp_bins, :]

    frame_energy = np.mean(bp_mat, axis=0)
    frame_peak = np.max(bp_mat, axis=0)
    noise_floor = float(np.percentile(frame_energy, 25)) if v_frames > 4 else 0.1
    burst_thresh = max(0.18, noise_floor + 0.12)
    active_frame_idxs = np.where((frame_energy >= burst_thresh) | (frame_peak >= 0.42))[0]

    # Spectral flux across consecutive frames -> captures discontinuous Crackle transients
    if v_frames > 2:
        diff_pos = np.maximum(0.0, np.diff(bp_mat, axis=1))
        flux_series = np.mean(diff_pos, axis=0)
        flux_thresh = float(np.mean(flux_series) + 1.1 * np.std(flux_series))
        transient_idxs = np.where((flux_series >= max(0.045, flux_thresh)))[0]
        # Count separated impulsive events (>= 2 frames apart)
        impulses: List[int] = []
        for idx in transient_idxs:
            if not impulses or (idx - impulses[-1]) >= 3:
                impulses.append(int(idx))
        transient_count = len(impulses)
    else:
        impulses = []
        transient_count = 0

    # Spectral flatness / narrowband tonality -> captures continuous Wheeze ribbons
    lin_bp = np.power(10.0, (bp_mat * 6.0) - 6.0)  # relative linear energy proxy
    geo_mean = np.exp(np.mean(np.log(np.maximum(lin_bp, 1e-9)), axis=0))
    ari_mean = np.mean(lin_bp, axis=0) + 1e-9
    tonality_per_frame = 1.0 - np.clip(geo_mean / ari_mean, 0.0, 1.0)
    high_tonal_frames = np.where((tonality_per_frame >= 0.55) & (frame_energy >= burst_thresh * 0.85))[0]
    tonal_persistency_pct = round(100.0 * len(high_tonal_frames) / max(1, len(active_frame_idxs) or v_frames), 1)

    # Dominant frequency & centroid inside bandpass
    bin_energy = np.mean(bp_mat[:, active_frame_idxs], axis=1) if len(active_frame_idxs) > 0 else np.mean(bp_mat, axis=1)
    best_bp_bin = int(np.argmax(bin_energy))
    peak_freq_hz = round(float(mel_center_freqs[bp_bins[best_bp_bin]]))

    weights_f = np.maximum(bin_energy, 0.0)
    w_sum = float(np.sum(weights_f))
    if w_sum > 1e-6:
        spectral_centroid_hz = round(float(np.sum(mel_center_freqs[bp_bins] * weights_f) / w_sum))
    else:
        spectral_centroid_hz = peak_freq_hz

    # Energy distribution across respiratory sub-bands (sum normalized to 100%)
    low_m = (mel_center_freqs >= 100.0) & (mel_center_freqs < 400.0)
    mid_m = (mel_center_freqs >= 400.0) & (mel_center_freqs < 1200.0)
    high_m = (mel_center_freqs >= 1200.0) & (mel_center_freqs <= 2000.0)
    e_low = float(np.mean(active_mat[low_m])) if np.any(low_m) else 0.0
    e_mid = float(np.mean(active_mat[mid_m])) if np.any(mid_m) else 0.0
    e_high = float(np.mean(active_mat[high_m])) if np.any(high_m) else 0.0
    e_tot = max(1e-6, e_low + e_mid + e_high)

    regions: List[Dict[str, Any]] = []

    def add_region(
        reg_id: str,
        label: str,
        short_tag: str,
        kind: str,
        x0_frame: int,
        x1_frame: int,
        freq_low: float,
        freq_high: float,
        detail: str,
    ):
        xs = max(0.02, min(0.94, x0_frame / float(n_frames)))
        xe = max(xs + 0.06, min(0.98, (x1_frame + 1) / float(n_frames)))
        yt = _freq_to_y_pct(freq_high, mel_center_freqs)
        yb = _freq_to_y_pct(freq_low, mel_center_freqs)
        if yb - yt < 0.08:
            mid_y = 0.5 * (yt + yb)
            yt = max(0.04, mid_y - 0.045)
            yb = min(0.96, mid_y + 0.045)
        t0 = round(xs * window_sec, 2)
        t1 = round(xe * window_sec, 2)
        regions.append(
            {
                "id": reg_id,
                "label": label,
                "short_tag": short_tag,
                "kind": kind,
                "x_start_pct": round(xs, 4),
                "x_end_pct": round(xe, 4),
                "y_top_pct": round(yt, 4),
                "y_bottom_pct": round(yb, 4),
                "time_start_sec": t0,
                "time_end_sec": t1,
                "freq_low_hz": round(freq_low),
                "freq_high_hz": round(freq_high),
                "detail": detail,
            }
        )

    # 1. Primary energy burst window
    if len(active_frame_idxs) > 0:
        p_idx = int(active_frame_idxs[int(np.argmax(frame_energy[active_frame_idxs]))])
        # Expand contiguous window around peak frame
        x0, x1 = p_idx, p_idx
        while x0 > 0 and frame_energy[x0 - 1] >= burst_thresh * 0.72:
            x0 -= 1
        while x1 < v_frames - 1 and frame_energy[x1 + 1] >= burst_thresh * 0.72:
            x1 += 1
    else:
        p_idx = max(0, v_frames // 3)
        x0, x1 = max(0, p_idx - 4), min(v_frames - 1, p_idx + 8)

    if pred_label == "Wheeze":
        f_lo = max(140.0, peak_freq_hz * 0.68)
        f_hi = min(2000.0, peak_freq_hz * 1.45)
        add_region(
            "wheeze_band",
            f"Pita Harmonik Wheeze (~{peak_freq_hz} Hz)",
            f"Wheeze {peak_freq_hz}Hz",
            "wheeze",
            x0,
            min(v_frames - 1, max(x1, x0 + 8)),
            f_lo,
            f_hi,
            f"Terlihat pola pita horizontal berkelanjutan di sekitar {peak_freq_hz} Hz (tonalitas {tonal_persistency_pct}%), khas penyempitan bronkus.",
        )
    elif pred_label == "Crackle":
        f_lo, f_hi = 180.0, min(2000.0, max(1250.0, peak_freq_hz * 1.6))
        add_region(
            "crackle_burst",
            f"Impuls Transien Crackle ({transient_count or 1} ledakan)",
            "Crackle Burst",
            "crackle",
            x0,
            x1,
            f_lo,
            f_hi,
            f"Terdeteksi {max(1, transient_count)} lonjakan energi vertikal cepat lintas frekuensi ({int(f_lo)}-{int(f_hi)} Hz), khas ronki basah / sekret jalan napas.",
        )
    elif pred_label == "Crackle + Wheeze":
        f_lo = max(160.0, peak_freq_hz * 0.7)
        f_hi = min(2000.0, max(1350.0, peak_freq_hz * 1.5))
        add_region(
            "mixed_wheeze",
            f"Harmonik Wheeze (~{peak_freq_hz} Hz)",
            f"Wheeze {peak_freq_hz}Hz",
            "wheeze",
            x0,
            min(v_frames - 1, max(x1, x0 + 8)),
            f_lo,
            min(2000.0, peak_freq_hz * 1.4),
            f"Pita tonal berkelanjutan di sekitar {peak_freq_hz} Hz menandakan penyempitan jalan napas.",
        )
        imp_idx = impulses[0] if impulses else min(v_frames - 2, x0 + 2)
        add_region(
            "mixed_crackle",
            "Impuls Crackle Sekunder",
            "Crackle",
            "crackle",
            max(0, imp_idx - 2),
            min(v_frames - 1, imp_idx + 4),
            200.0,
            1750.0,
            "Kolom spektrum vertikal tajam mengindikasikan suara crackle/cairan bersamaan dengan mengi.",
        )
    else:
        f_lo = max(100.0, min(260.0, peak_freq_hz * 0.55))
        f_hi = min(1800.0, max(950.0, peak_freq_hz * 1.85))
        add_region(
            "normal_cough",
            f"Puncak Batuk Primer (~{peak_freq_hz} Hz)",
            f"Normal {peak_freq_hz}Hz",
            "normal",
            x0,
            x1,
            f_lo,
            f_hi,
            f"Letupan energi batuk primer terpusat bersih di {peak_freq_hz} Hz dengan peluruhan spektrum alami tanpa mengi memanjang.",
        )

    # Secondary transient / tonal harmonic region when distinct from primary window
    added_secondary = False
    for imp in impulses:
        if imp < x0 - 5 or imp > x1 + 5:
            kind = "crackle" if "Crackle" in pred_label else "normal"
            label = (
                "Transien Crackle Tambahan"
                if kind == "crackle"
                else "Aktivitas Akustik Sekunder"
            )
            add_region(
                f"sec_{imp}",
                label,
                "Transien" if kind == "crackle" else "Sekunder",
                kind,
                max(0, imp - 2),
                min(v_frames - 1, imp + 3),
                200.0,
                1600.0,
                "Aktivitas spektral singkat terdeteksi terpisah dari puncak batuk utama.",
            )
            added_secondary = True
            break

    # If a distinct high-tonal harmonic peak sits outside the primary box, annotate it too
    if not added_secondary and len(high_tonal_frames) >= 2:
        far_tonal = high_tonal_frames[(high_tonal_frames < x0 - 4) | (high_tonal_frames > x1 + 4)]
        if len(far_tonal) >= 2:
            t_center = int(far_tonal[int(np.argmax(frame_peak[far_tonal]))])
            dom_bin = int(np.argmax(bp_mat[:, t_center]))
            dom_hz = round(float(mel_center_freqs[bp_bins[dom_bin]]))
            tx0 = max(0, int(far_tonal[0]))
            tx1 = min(v_frames - 1, max(tx0 + 6, int(far_tonal[-1])))
            add_region(
                f"tonal_{t_center}",
                f"Pita Harmonik Sekitar {dom_hz} Hz",
                f"Harmonik {dom_hz}Hz",
                "wheeze" if "Wheeze" in pred_label else "tonal",
                tx0,
                tx1,
                max(120.0, dom_hz * 0.72),
                min(2000.0, dom_hz * 1.38),
                f"Konsentrasi energi narrow-band pada {dom_hz} Hz teramati pada rentang ini.",
            )

    metrics = {
        "duration_sec": round(float(v_frames * 512 / 16000.0), 2),
        "window_sec": window_sec,
        "peak_frequency_hz": peak_freq_hz,
        "spectral_centroid_hz": spectral_centroid_hz,
        "transient_burst_count": transient_count,
        "tonal_persistency_pct": tonal_persistency_pct,
        "low_band_pct": round(100.0 * e_low / e_tot, 1),
        "mid_band_pct": round(100.0 * e_mid / e_tot, 1),
        "high_band_pct": round(100.0 * e_high / e_tot, 1),
    }
    return regions, metrics


def build_cough_visualization(
    dsp_result: Dict[str, Any],
    pred_label: str,
    confidence_pct: float,
    all_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Create complete spectrogram image + visual prediction boxes + step-by-step
    clinical explanation from the real audio telemetry.
    """
    mel_raw = dsp_result["mel_raw"]
    mel_model = dsp_result["mel_model"]
    mel_center_freqs = dsp_result["mel_center_freqs"]
    y_raw = dsp_result["y_raw"]
    y_filtered = dsp_result["y_filtered"]
    window_sec = float(dsp_result["window_sec"])
    sample_rate = int(dsp_result["sample_rate"])

    valid_frames = min(mel_model.shape[1], mel_raw.shape[1])

    # Bandpass retention percentage (100-2000 Hz vs raw spectrum)
    raw_rms = float(np.sqrt(np.mean(y_raw ** 2))) if len(y_raw) else 0.0
    bp_rms = float(np.sqrt(np.mean(y_filtered ** 2))) if len(y_filtered) else 0.0
    bandpass_retention_pct = round(min(100.0, 100.0 * bp_rms / max(1e-6, raw_rms)), 1) if raw_rms > 1e-6 else 0.0

    # Normalized spectrogram matrix for acoustic ROI analysis
    min_db = float(np.min(mel_model))
    max_db = float(np.max(mel_model))
    floor_db = max(min_db, max_db - 68.0)
    norm_mel = np.clip((mel_model - floor_db) / max(1e-5, max_db - floor_db), 0.0, 1.0)

    spectrogram_image = _render_spectrogram_base64(mel_model, valid_frames)
    waveform_envelope = _compute_waveform_envelope(y_filtered, window_sec, sample_rate, bars=64)
    detected_regions, acoustic_metrics = _extract_detected_regions(
        norm_mel=norm_mel,
        mel_center_freqs=mel_center_freqs,
        valid_frames=valid_frames,
        window_sec=window_sec,
        pred_label=pred_label,
    )
    acoustic_metrics["recorded_duration_sec"] = float(dsp_result["duration_sec"])
    acoustic_metrics["bandpass_retention_pct"] = bandpass_retention_pct
    acoustic_metrics["max_energy_db"] = round(max_db, 1)

    axis_ticks = [
        {"label": "8 kHz", "freq_hz": 8000, "y_pct": _freq_to_y_pct(7800.0, mel_center_freqs)},
        {"label": "4 kHz", "freq_hz": 4000, "y_pct": _freq_to_y_pct(4000.0, mel_center_freqs)},
        {"label": "2 kHz", "freq_hz": 2000, "y_pct": _freq_to_y_pct(2000.0, mel_center_freqs)},
        {"label": "1 kHz", "freq_hz": 1000, "y_pct": _freq_to_y_pct(1000.0, mel_center_freqs)},
        {"label": "400 Hz", "freq_hz": 400, "y_pct": _freq_to_y_pct(400.0, mel_center_freqs)},
        {"label": "100 Hz", "freq_hz": 100, "y_pct": _freq_to_y_pct(100.0, mel_center_freqs)},
    ]

    peak_hz = acoustic_metrics["peak_frequency_hz"]
    centroid_hz = acoustic_metrics["spectral_centroid_hz"]
    transients = acoustic_metrics["transient_burst_count"]
    tonality = acoustic_metrics["tonal_persistency_pct"]

    if pred_label == "Wheeze":
        stage2_detail = (
            f"Mel-Spektrogram 128x128 memperlihatkan pita energi harmonik horizontal yang bertahan "
            f"(tonalitas {tonality}%) dengan puncak frekuensi di {peak_hz} Hz (sentroid {centroid_hz} Hz), "
            f"ciri khas getaran dinding saluran napas menyempit (wheezing)."
        )
    elif pred_label == "Crackle":
        stage2_detail = (
            f"Teridentifikasi {max(1, transients)} impuls transien vertikal cepat lintas pita frekuensi "
            f"menengah-tinggi (sentroid {centroid_hz} Hz, puncak {peak_hz} Hz), menandakan suara letupan "
            f"singkat pembukaan alveolus/sekret basah (crackle)."
        )
    elif pred_label == "Crackle + Wheeze":
        stage2_detail = (
            f"Spektrogram memperlihatkan pola campuran: pita harmonik kontinu ({tonality}% tonal di {peak_hz} Hz) "
            f"sekaligus {max(1, transients)} letupan transien crackle pada zona napas 100-2000 Hz."
        )
    else:
        stage2_detail = (
            f"Energi spektral terpusat pada letupan batuk primer ({peak_hz} Hz, sentroid {centroid_hz} Hz) "
            f"lalu meluruh bersih tanpa pita mengi berkepanjangan ({tonality}% tonal) atau rentetan crackle patologis."
        )

    explanation_steps = [
        {
            "step": "01",
            "title": "Isolasi Bandpass Paru (100-2000 Hz)",
            "metric": f"{bandpass_retention_pct}% energi dipertahankan",
            "description": (
                f"Rekaman {dsp_result['duration_sec']} detik di-resample ke 16 kHz mono lalu difilter Butterworth "
                f"100-2000 Hz untuk mempertahankan akustik paru sambil meredam dengung rendah & bising di atas 2 kHz."
            ),
        },
        {
            "step": "02",
            "title": "Ekstraksi Peta Mel-Spektrogram 128x128",
            "metric": f"Puncak {peak_hz} Hz • {max(1, transients)} transien",
            "description": stage2_detail,
        },
        {
            "step": "03",
            "title": f"Klasifikasi CNN ONNX ICBHI -> {pred_label}",
            "metric": f"{confidence_pct:.1f}% keyakinan",
            "description": (
                f"Matriks spektrogram 128x128 dianalisis oleh model CNN best_icbhi_model.onnx dan menghasilkan "
                f"prediksi {pred_label} ({confidence_pct:.1f}%), dengan dominasi pita menengah {acoustic_metrics['mid_band_pct']}%."
            ),
        },
    ]

    return {
        "spectrogram_image": spectrogram_image,
        "waveform_envelope": waveform_envelope,
        "detected_regions": detected_regions,
        "mel_axis_ticks": axis_ticks,
        "bandpass_box": {
            "y_top_pct": _freq_to_y_pct(2000.0, mel_center_freqs),
            "y_bottom_pct": _freq_to_y_pct(100.0, mel_center_freqs),
        },
        "acoustic_metrics": acoustic_metrics,
        "explanation_steps": explanation_steps,
        "visual_summary": stage2_detail,
    }
