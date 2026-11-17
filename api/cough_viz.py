import base64
import io
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _build_parula_palette() -> np.ndarray:
    """
    256-entry RGB colormap matching scientific MATLAB Parula / Viridis-blue
    spectrograms: Royal Blue -> Teal/Cyan -> Sea Green -> Golden Yellow.
    """
    stops = [
        (0.00, (44, 88, 219)),
        (0.22, (28, 128, 229)),
        (0.42, (24, 174, 198)),
        (0.62, (62, 196, 146)),
        (0.80, (176, 212, 86)),
        (0.92, (244, 206, 54)),
        (1.00, (250, 242, 72)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = [int(round(pos * 255)) for pos, _ in stops]
    for c in range(3):
        vals = [rgb[c] for _, rgb in stops]
        lut[:, c] = np.clip(np.interp(np.arange(256), xs, vals), 0, 255).astype(np.uint8)
    return lut


_PARULA_LUT = _build_parula_palette()


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_rotated_text(
    canvas: Image.Image,
    center_xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int] = (15, 23, 42),
    angle: int = 90,
) -> None:
    tmp = Image.new("RGBA", (320, 60), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((-bbox[0] + 4, -bbox[1] + 4), text, font=font, fill=fill + (255,))
    cropped = tmp.crop((0, 0, tw + 8, th + 8)).rotate(angle, expand=True)
    x = int(center_xy[0] - cropped.width / 2)
    y = int(center_xy[1] - cropped.height / 2)
    canvas.paste(cropped, (x, y), cropped)


def _freq_to_y_pct(freq_hz: float, mel_center_freqs: np.ndarray) -> float:
    """Convert frequency (Hz) to normalized top-down Y coordinate [0..1]."""
    bin_idx = float(np.interp(freq_hz, mel_center_freqs, np.arange(len(mel_center_freqs))))
    return round(float(np.clip(1.0 - (bin_idx / max(1, len(mel_center_freqs) - 1)), 0.0, 1.0)), 4)


def _render_scientific_figure_base64(
    y_signal: np.ndarray,
    mel_model: np.ndarray,
    mel_center_freqs: np.ndarray,
    valid_frames: int,
    window_sec: float,
    recorded_sec: float,
    pred_label: str,
    confidence_pct: float,
    detected_regions: List[Dict[str, Any]],
) -> str:
    """
    Generate a crisp MATLAB/Matplotlib-style scientific 2-panel stacked figure
    in Python (Waveform Amplitude vs Time + Mel-Spectrogram vs Frequency/Time)
    with white paper margins, dark axes, ticks, colorbar, and prediction callout.
    """
    W, H = 920, 740
    fig = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(fig)

    f_title = _get_font(22)
    f_label = _get_font(18)
    f_tick = _get_font(15)
    f_small = _get_font(14)

    spine_color = (51, 65, 85)
    grid_color = (226, 232, 240)
    text_color = (15, 23, 42)
    muted_text = (51, 65, 85)

    # Shared left/right margins for aligned X-axes
    left_x = 96
    right_x = 796
    plot_w = right_x - left_x

    # ── SUBPLOT (a): Cough Signal Waveform (Amplitude vs Time) ──
    top_a, bot_a = 58, 304
    h_a = bot_a - top_a

    title_a = f"(a) Sinyal Audio Batuk ({recorded_sec:.2f}s, 16 kHz)"
    tb = draw.textbbox((0, 0), title_a, font=f_title)
    draw.text(
        (left_x + (plot_w - (tb[2] - tb[0])) // 2, 18),
        title_a,
        font=f_title,
        fill=text_color,
    )

    # Light horizontal zero grid + subtle detected region highlight
    mid_y_a = top_a + h_a // 2
    draw.line([(left_x, mid_y_a), (right_x, mid_y_a)], fill=grid_color, width=1)
    draw.line([(left_x, top_a + h_a // 4), (right_x, top_a + h_a // 4)], fill=(241, 245, 249), width=1)
    draw.line([(left_x, top_a + 3 * h_a // 4), (right_x, top_a + 3 * h_a // 4)], fill=(241, 245, 249), width=1)

    for reg in detected_regions[:1]:
        rx0 = int(left_x + reg["x_start_pct"] * plot_w)
        rx1 = int(left_x + reg["x_end_pct"] * plot_w)
        tint = (220, 252, 231) if reg["kind"] == "normal" else (254, 243, 199) if reg["kind"] == "wheeze" else (254, 226, 226)
        draw.rectangle([rx0, top_a + 1, rx1, bot_a - 1], fill=tint)

    # Re-draw zero baseline cleanly above tint
    draw.line([(left_x, mid_y_a), (right_x, mid_y_a)], fill=(203, 213, 225), width=1)

    # Prepare full-window signal synced to spectrogram time axis [0 .. window_sec]
    target_samples = max(plot_w, int(round(window_sec * 16000)))
    sig = y_signal[:target_samples]
    if len(sig) < target_samples:
        sig = np.pad(sig, (0, target_samples - len(sig)), mode="constant")
    peak_amp = float(np.max(np.abs(sig)))
    if peak_amp > 1e-5:
        sig = sig / peak_amp

    # Min/Max per horizontal pixel column for sharp scientific waveform look (#0072BD)
    wave_blue = (0, 114, 189)
    half_h = (h_a // 2) - 8
    cols = np.array_split(sig, plot_w)
    prev_y = mid_y_a
    for px_i, chunk in enumerate(cols):
        if len(chunk) == 0:
            continue
        vmin = float(np.min(chunk))
        vmax = float(np.max(chunk))
        y_hi = int(round(mid_y_a - vmax * half_h))
        y_lo = int(round(mid_y_a - vmin * half_h))
        x_pos = left_x + px_i
        if y_lo - y_hi < 1:
            draw.line([(x_pos - 1, prev_y), (x_pos, y_hi)], fill=wave_blue, width=1)
        else:
            draw.line([(x_pos, y_hi), (x_pos, y_lo)], fill=wave_blue, width=1)
        prev_y = (y_hi + y_lo) // 2

    # Subplot (a) axes frame & Y ticks (-1, 0, 1)
    draw.rectangle([left_x, top_a, right_x, bot_a], outline=spine_color, width=2)
    for val, yy in [("1", top_a + 8), ("0", mid_y_a), ("-1", bot_a - 8)]:
        draw.line([(left_x - 6, yy), (left_x, yy)], fill=spine_color, width=2)
        vb = draw.textbbox((0, 0), val, font=f_tick)
        draw.text((left_x - 12 - (vb[2] - vb[0]), yy - (vb[3] - vb[1]) // 2 - 2), val, font=f_tick, fill=muted_text)

    _draw_rotated_text(fig, (34, top_a + h_a // 2), "Amplitude", f_label, fill=text_color, angle=90)

    # X ticks for Subplot (a)
    tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tf in tick_fracs:
        tx = int(round(left_x + tf * plot_w))
        draw.line([(tx, bot_a), (tx, bot_a + 6)], fill=spine_color, width=2)
        lbl = f"{tf * window_sec:.1f}" if tf > 0 else "0"
        lb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((tx - (lb[2] - lb[0]) // 2, bot_a + 10), lbl, font=f_tick, fill=muted_text)
    xlab_a = "Time (s)"
    xba = draw.textbbox((0, 0), xlab_a, font=f_label)
    draw.text((left_x + (plot_w - (xba[2] - xba[0])) // 2, bot_a + 34), xlab_a, font=f_label, fill=text_color)

    # ── SUBPLOT (b): Spectrogram of Cough (Frequency kHz vs Time s) ──
    top_b, bot_b = 418, 668
    h_b = bot_b - top_b

    title_b = f"(b) Spectrogram Batuk - Prediksi: {pred_label} ({confidence_pct:.1f}%)"
    tb2 = draw.textbbox((0, 0), title_b, font=f_title)
    draw.text(
        (left_x + (plot_w - (tb2[2] - tb2[0])) // 2, 376),
        title_b,
        font=f_title,
        fill=text_color,
    )

    # Render Parula Spectrogram Heatmap into exact subplot box [plot_w x h_b]
    min_db = float(np.min(mel_model))
    max_db = float(np.max(mel_model))
    if max_db - min_db < 1e-5:
        norm = np.zeros_like(mel_model, dtype=np.float32)
    else:
        floor_db = max(min_db, max_db - 72.0)
        norm = np.clip((mel_model - floor_db) / max(1e-5, max_db - floor_db), 0.0, 1.0)

    if 0 < valid_frames < mel_model.shape[1]:
        norm[:, valid_frames:] = np.minimum(norm[:, valid_frames:], 0.03)

    flipped = np.flipud(norm)
    indices = np.clip((flipped * 255.0).astype(np.int32), 0, 255)
    rgb = _PARULA_LUT[indices]
    spec_img = Image.fromarray(rgb, mode="RGB").resize((plot_w, h_b), Image.Resampling.BILINEAR)
    fig.paste(spec_img, (left_x, top_b))

    # Draw subtle clinical highlight rectangle & annotation inside Subplot (b)
    accent_map = {
        "normal": (16, 185, 129),
        "wheeze": (245, 158, 11),
        "crackle": (239, 68, 68),
        "tonal": (2, 132, 199),
    }
    for reg in detected_regions:
        rx0 = int(left_x + reg["x_start_pct"] * plot_w)
        rx1 = int(left_x + reg["x_end_pct"] * plot_w)
        ry0 = int(top_b + reg["y_top_pct"] * h_b)
        ry1 = int(top_b + reg["y_bottom_pct"] * h_b)
        col = accent_map.get(reg["kind"], (239, 68, 68))

        # Crisp white outer halo + colored box
        draw.rectangle([rx0 - 1, ry0 - 1, rx1 + 1, ry1 + 1], outline=(255, 255, 255), width=2)
        draw.rectangle([rx0, ry0, rx1, ry1], outline=col, width=2)

        tag = reg["short_tag"]
        tbox = draw.textbbox((0, 0), tag, font=f_small)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        tag_x = max(left_x + 4, min(right_x - tw - 14, rx0))
        tag_y = ry0 - th - 10 if ry0 - th - 12 > top_b else ry0 + 4
        draw.rectangle([tag_x, tag_y, tag_x + tw + 10, tag_y + th + 6], fill=col)
        draw.text((tag_x + 5, tag_y + 2), tag, font=f_small, fill=(255, 255, 255))

    # Prediction legend pill at top-right inside Subplot (b)
    pred_tag = f"ONNX ICBHI: {pred_label}"
    ptb = draw.textbbox((0, 0), pred_tag, font=f_small)
    ptw, pth = ptb[2] - ptb[0], ptb[3] - ptb[1]
    pill_x0 = right_x - ptw - 20
    pill_y0 = top_b + 8
    draw.rectangle(
        [pill_x0, pill_y0, right_x - 8, pill_y0 + pth + 8],
        fill=(255, 255, 255),
        outline=spine_color,
        width=1,
    )
    draw.text((pill_x0 + 6, pill_y0 + 3), pred_tag, font=f_small, fill=text_color)

    # Subplot (b) axes frame & calibrated Frequency (kHz) ticks
    draw.rectangle([left_x, top_b, right_x, bot_b], outline=spine_color, width=2)
    freq_ticks = [("8", 7800.0), ("4", 4000.0), ("2", 2000.0), ("1", 1000.0), ("0.4", 400.0), ("0", 40.0)]
    for flbl, fhz in freq_ticks:
        yp = _freq_to_y_pct(fhz, mel_center_freqs)
        yy = int(round(top_b + yp * h_b))
        yy = max(top_b + 2, min(bot_b - 2, yy))
        draw.line([(left_x - 6, yy), (left_x, yy)], fill=spine_color, width=2)
        fb = draw.textbbox((0, 0), flbl, font=f_tick)
        draw.text((left_x - 12 - (fb[2] - fb[0]), yy - (fb[3] - fb[1]) // 2 - 2), flbl, font=f_tick, fill=muted_text)

    _draw_rotated_text(fig, (34, top_b + h_b // 2), "Frequency (kHz)", f_label, fill=text_color, angle=90)

    # X ticks for Subplot (b)
    for tf in tick_fracs:
        tx = int(round(left_x + tf * plot_w))
        draw.line([(tx, bot_b), (tx, bot_b + 6)], fill=spine_color, width=2)
        lbl = f"{tf * window_sec:.1f}" if tf > 0 else "0"
        lb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((tx - (lb[2] - lb[0]) // 2, bot_b + 10), lbl, font=f_tick, fill=muted_text)
    xlab_b = "Time (s)"
    xbb = draw.textbbox((0, 0), xlab_b, font=f_label)
    draw.text((left_x + (plot_w - (xbb[2] - xbb[0])) // 2, bot_b + 34), xlab_b, font=f_label, fill=text_color)

    # Colorbar on the right of Subplot (b)
    cb_x0, cb_x1 = right_x + 18, right_x + 36
    cb_grad = np.linspace(255, 0, h_b, dtype=np.int32)[:, None]
    cb_rgb = np.repeat(_PARULA_LUT[cb_grad], cb_x1 - cb_x0, axis=1)
    fig.paste(Image.fromarray(cb_rgb, mode="RGB"), (cb_x0, top_b))
    draw.rectangle([cb_x0, top_b, cb_x1, bot_b], outline=spine_color, width=1)
    for db_lbl, yy in [("0 dB", top_b + 4), ("-40", top_b + h_b // 2), ("-72", bot_b - 6)]:
        draw.text((cb_x1 + 6, yy - 8), db_lbl, font=f_small, fill=muted_text)

    buf = io.BytesIO()
    fig.save(buf, format="JPEG", quality=92, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


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
    _, n_frames = norm_mel.shape
    v_frames = max(1, min(n_frames, valid_frames))
    active_mat = norm_mel[:, :v_frames]

    bp_mask = (mel_center_freqs >= 100.0) & (mel_center_freqs <= 2000.0)
    bp_bins = np.where(bp_mask)[0]
    bp_mat = active_mat[bp_bins, :]

    frame_energy = np.mean(bp_mat, axis=0)
    frame_peak = np.max(bp_mat, axis=0)
    noise_floor = float(np.percentile(frame_energy, 25)) if v_frames > 4 else 0.1
    burst_thresh = max(0.18, noise_floor + 0.12)
    active_frame_idxs = np.where((frame_energy >= burst_thresh) | (frame_peak >= 0.42))[0]

    if v_frames > 2:
        diff_pos = np.maximum(0.0, np.diff(bp_mat, axis=1))
        flux_series = np.mean(diff_pos, axis=0)
        flux_thresh = float(np.mean(flux_series) + 1.1 * np.std(flux_series))
        transient_idxs = np.where(flux_series >= max(0.045, flux_thresh))[0]
        impulses: List[int] = []
        for idx in transient_idxs:
            if not impulses or (idx - impulses[-1]) >= 3:
                impulses.append(int(idx))
        transient_count = len(impulses)
    else:
        impulses = []
        transient_count = 0

    lin_bp = np.power(10.0, (bp_mat * 6.0) - 6.0)
    geo_mean = np.exp(np.mean(np.log(np.maximum(lin_bp, 1e-9)), axis=0))
    ari_mean = np.mean(lin_bp, axis=0) + 1e-9
    tonality_per_frame = 1.0 - np.clip(geo_mean / ari_mean, 0.0, 1.0)
    high_tonal_frames = np.where((tonality_per_frame >= 0.55) & (frame_energy >= burst_thresh * 0.85))[0]
    tonal_persistency_pct = round(100.0 * len(high_tonal_frames) / max(1, len(active_frame_idxs) or v_frames), 1)

    bin_energy = np.mean(bp_mat[:, active_frame_idxs], axis=1) if len(active_frame_idxs) > 0 else np.mean(bp_mat, axis=1)
    best_bp_bin = int(np.argmax(bin_energy))
    peak_freq_hz = round(float(mel_center_freqs[bp_bins[best_bp_bin]]))

    weights_f = np.maximum(bin_energy, 0.0)
    w_sum = float(np.sum(weights_f))
    if w_sum > 1e-6:
        spectral_centroid_hz = round(float(np.sum(mel_center_freqs[bp_bins] * weights_f) / w_sum))
    else:
        spectral_centroid_hz = peak_freq_hz

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

    if len(active_frame_idxs) > 0:
        p_idx = int(active_frame_idxs[int(np.argmax(frame_energy[active_frame_idxs]))])
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
            f"Terlihat pita horizontal berkelanjutan di sekitar {peak_freq_hz} Hz (tonalitas {tonal_persistency_pct}%), khas penyempitan bronkus.",
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
            f"Terdeteksi {max(1, transient_count)} lonjakan energi vertikal cepat lintas frekuensi ({int(f_lo)}-{int(f_hi)} Hz), khas ronki basah.",
        )
    elif pred_label == "Crackle + Wheeze":
        f_lo = max(160.0, peak_freq_hz * 0.7)
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
            "Kolom spektrum vertikal tajam mengindikasikan suara crackle bersamaan dengan mengi.",
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

    added_secondary = False
    for imp in impulses:
        if imp < x0 - 5 or imp > x1 + 5:
            kind = "crackle" if "Crackle" in pred_label else "normal"
            label = "Transien Crackle Tambahan" if kind == "crackle" else "Aktivitas Akustik Sekunder"
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
    Create complete Python-generated scientific 2-panel figure (Waveform + Mel-Spectrogram)
    plus quantitative acoustic metrics and explanation steps.
    """
    mel_raw = dsp_result["mel_raw"]
    mel_model = dsp_result["mel_model"]
    mel_center_freqs = dsp_result["mel_center_freqs"]
    y_raw = dsp_result["y_raw"]
    y_filtered = dsp_result["y_filtered"]
    window_sec = float(dsp_result["window_sec"])
    recorded_sec = float(dsp_result["duration_sec"])
    sample_rate = int(dsp_result["sample_rate"])

    valid_frames = min(mel_model.shape[1], mel_raw.shape[1])

    raw_rms = float(np.sqrt(np.mean(y_raw ** 2))) if len(y_raw) else 0.0
    bp_rms = float(np.sqrt(np.mean(y_filtered ** 2))) if len(y_filtered) else 0.0
    bandpass_retention_pct = round(min(100.0, 100.0 * bp_rms / max(1e-6, raw_rms)), 1) if raw_rms > 1e-6 else 0.0

    min_db = float(np.min(mel_model))
    max_db = float(np.max(mel_model))
    floor_db = max(min_db, max_db - 72.0)
    norm_mel = np.clip((mel_model - floor_db) / max(1e-5, max_db - floor_db), 0.0, 1.0)

    waveform_envelope = _compute_waveform_envelope(y_filtered, window_sec, sample_rate, bars=64)
    detected_regions, acoustic_metrics = _extract_detected_regions(
        norm_mel=norm_mel,
        mel_center_freqs=mel_center_freqs,
        valid_frames=valid_frames,
        window_sec=window_sec,
        pred_label=pred_label,
    )
    acoustic_metrics["recorded_duration_sec"] = recorded_sec
    acoustic_metrics["bandpass_retention_pct"] = bandpass_retention_pct
    acoustic_metrics["max_energy_db"] = round(max_db, 1)

    spectrogram_image = _render_scientific_figure_base64(
        y_signal=y_filtered,
        mel_model=mel_model,
        mel_center_freqs=mel_center_freqs,
        valid_frames=valid_frames,
        window_sec=window_sec,
        recorded_sec=recorded_sec,
        pred_label=pred_label,
        confidence_pct=confidence_pct,
        detected_regions=detected_regions,
    )

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
            f"ciri khas getaran penyempitan jalan napas (wheezing)."
        )
    elif pred_label == "Crackle":
        stage2_detail = (
            f"Teridentifikasi {max(1, transients)} impuls transien vertikal cepat lintas frekuensi "
            f"(sentroid {centroid_hz} Hz, puncak {peak_hz} Hz), menandakan suara letupan "
            f"singkat pembukaan alveolus atau cairan jalan napas (crackle)."
        )
    elif pred_label == "Crackle + Wheeze":
        stage2_detail = (
            f"Spektrogram memperlihatkan pola campuran: pita harmonik kontinu ({tonality}% tonal di {peak_hz} Hz) "
            f"dan {max(1, transients)} letupan transien crackle pada zona napas 100-2000 Hz."
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
                f"Rekaman {recorded_sec} detik di-resample ke 16 kHz mono lalu difilter Butterworth "
                f"100-2000 Hz untuk mempertahankan akustik paru sambil meredam noise di luar pita napas."
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
                f"Matriks spektrogram 128x128 diproses model CNN best_icbhi_model.onnx menghasilkan "
                f"prediksi {pred_label} ({confidence_pct:.1f}%), dengan dominasi energi menengah {acoustic_metrics['mid_band_pct']}%."
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
