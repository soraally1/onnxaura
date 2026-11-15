import base64
import io
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _gaussian_blur_2d(arr2d: np.ndarray, sigma: float) -> np.ndarray:
    """Fast C-backed 2D Gaussian blur via Pillow 'L' mode, avoiding SciPy."""
    a = arr2d.astype(np.float32)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo + 1e-6:
        return a
    u8 = np.clip(((a - lo) / (hi - lo)) * 255.0, 0.0, 255.0).astype(np.uint8)
    blurred_u8 = np.array(
        Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma))),
        dtype=np.float32,
    )
    return lo + (blurred_u8 / 255.0) * (hi - lo)


def _gaussian_blur_rgb(arr3d: np.ndarray, sigma: float) -> np.ndarray:
    """Fast RGB Gaussian blur via Pillow."""
    img = Image.fromarray(np.clip(arr3d, 0, 255).astype(np.uint8), mode="RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.array(blurred, dtype=np.float32)


def _grey_closing_2d(arr2d: np.ndarray, size: int = 5) -> np.ndarray:
    """Morphological grayscale closing (dilation -> erosion) via Pillow C filters."""
    u8 = np.clip(arr2d, 0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    closed = img.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.array(closed, dtype=np.float32)


def _encode_jpeg_data_url(image: Image.Image, max_width: int = 420, quality: int = 84) -> str:
    """Encodes a PIL Image as a compact JPEG data URL for fast mobile transfer."""
    img = image.convert("RGB")
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_h = max(1, int(round(img.height * ratio)))
        img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_lighting_and_dull_razor(rgb: np.ndarray) -> np.ndarray:
    """
    Applies gentle gray-world illumination constancy and morphological
    Dull Razor suppression to reduce stray hair/specular glare false positives.
    """
    arr = rgb.astype(np.float32)

    means = np.mean(arr, axis=(0, 1))
    means = np.maximum(means, 1.0)
    gray_target = float(np.mean(means))
    scales = np.clip(gray_target / means, 0.80, 1.25)
    balanced = np.clip(arr * scales[None, None, :], 0.0, 255.0)
    constancy = 0.60 * arr + 0.40 * balanced

    lum = 0.299 * constancy[:, :, 0] + 0.587 * constancy[:, :, 1] + 0.114 * constancy[:, :, 2]
    closed_lum = _grey_closing_2d(lum, size=5)
    hair_response = closed_lum - lum

    sat = (np.max(constancy, axis=2) - np.min(constancy, axis=2)) / (np.max(constancy, axis=2) + 1e-5)
    artifact_mask = (hair_response > 15.0) | ((lum > 242.0) & (sat < 0.12))
    artifact_weight = _gaussian_blur_2d(artifact_mask.astype(np.float32), sigma=1.1)
    artifact_weight = np.clip(artifact_weight * 1.3, 0.0, 0.80)[:, :, None]

    smoothed = _gaussian_blur_rgb(constancy, sigma=1.4)
    cleaned = constancy * (1.0 - artifact_weight) + smoothed * artifact_weight
    return np.clip(cleaned, 0.0, 255.0)


def _compute_skin_probability(rgb: np.ndarray) -> np.ndarray:
    """
    Computes continuous [0..1] soft skin probability in YCbCr + warm RGB space
    across all Fitzpatrick skin tones without harsh binary dropouts.
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Gaussian-shaped YCbCr skin likelihood centered on human skin locus
    cr_score = np.exp(-0.5 * ((cr - 149.0) / 14.0) ** 2)
    cb_score = np.exp(-0.5 * ((cb - 106.0) / 15.0) ** 2)
    warm_score = np.clip((r - b + 8.0) / 36.0, 0.0, 1.0)
    lum_ok = np.clip((lum - 24.0) / 28.0, 0.0, 1.0)

    prob = cr_score * cb_score * (0.45 + 0.55 * warm_score) * lum_ok
    return _gaussian_blur_2d(prob, sigma=2.5)


def _sample_patch(arr: np.ndarray, cy: int, cx: int, ry: int, rx: int) -> float:
    h, w = arr.shape
    y0 = max(0, cy - ry)
    y1 = min(h, cy + ry + 1)
    x0 = max(0, cx - rx)
    x1 = min(w, cx + rx + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(np.mean(arr[y0:y1, x0:x1]))


def _detect_face_geometry(rgb: np.ndarray, skin_prob: np.ndarray) -> Dict[str, Any]:
    """
    Hybrid Viola-Jones Haar-like Eye-Pair + YCbCr Skin-Core face localization.
    Locks strictly onto the eyes, nasal bridge, cheeks, mouth, and jaw without
    drifting onto warm background walls or shoulders/clothing.
    """
    h, w, _ = rgb.shape
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    cr = 128.0 + 0.5 * rgb[:, :, 0] - 0.418688 * rgb[:, :, 1] - 0.081312 * rgb[:, :, 2]

    # Work on compact grid (~120px wide) for ultra-fast multi-scale eye-pair search
    scale = 120.0 / float(max(w, 1))
    sw = max(48, int(round(w * scale)))
    sh = max(48, int(round(h * scale)))

    lum_s = np.array(
        Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L").resize((sw, sh), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    skin_s = np.array(
        Image.fromarray(np.clip(skin_prob * 255.0, 0, 255).astype(np.uint8), mode="L").resize((sw, sh), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # Integral images for O(1) rectangular mean lookups
    ii_lum = np.pad(lum_s, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    ii_skin = np.pad(skin_s, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    def box_mean(ii: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
        y0c = max(0, min(sh - 1, y0))
        y1c = max(y0c + 1, min(sh, y1))
        x0c = max(0, min(sw - 1, x0))
        x1c = max(x0c + 1, min(sw, x1))
        area = float((y1c - y0c) * (x1c - x0c))
        total = ii[y1c, x1c] - ii[y0c, x1c] - ii[y1c, x0c] + ii[y0c, x0c]
        return float(total / area)

    best_score = -1e9
    best_cfg = None

    # Search eye line (14%..64% height), center X (18%..82% width), half-IPD (9.5%..24.5% width)
    for half_ipd_norm in (0.095, 0.118, 0.145, 0.175, 0.21, 0.245):
        d = max(5, int(round(half_ipd_norm * sw)))
        er_y = max(2, int(round(d * 0.28)))
        er_x = max(2, int(round(d * 0.34)))

        cy_min = max(int(sh * 0.14), int(d * 0.75))
        cy_max = min(int(sh * 0.64), sh - int(d * 1.25))
        cx_min = max(int(sw * 0.18), int(d * 0.95))
        cx_max = min(int(sw * 0.82), sw - int(d * 0.95))

        for ey in range(cy_min, max(cy_min + 1, cy_max), 2):
            for cx in range(cx_min, max(cx_min + 1, cx_max), 2):
                # Require forehead, both orbital areas, nose bridge, and cheeks inside skin region
                sk_fh = box_mean(ii_skin, ey - int(d * 0.90), ey - int(d * 0.30), cx - int(d * 0.7), cx + int(d * 0.7))
                sk_eye_l = box_mean(ii_skin, ey - er_y, ey + int(d * 0.55), cx - d - int(d * 0.55), cx - d + int(d * 0.35))
                sk_eye_r = box_mean(ii_skin, ey - er_y, ey + int(d * 0.55), cx + d - int(d * 0.35), cx + d + int(d * 0.55))
                sk_bridge = box_mean(ii_skin, ey - er_y, ey + int(d * 0.55), cx - max(2, d // 3), cx + max(2, d // 3))
                sk_ch = box_mean(ii_skin, ey + int(d * 0.35), ey + int(d * 1.05), cx - int(d * 0.95), cx + int(d * 0.95))
                if sk_fh < 0.18 or sk_eye_l < 0.20 or sk_eye_r < 0.20 or sk_bridge < 0.20 or sk_ch < 0.18:
                    continue

                lum_le = box_mean(ii_lum, ey - er_y, ey + er_y, cx - d - er_x, cx - d + er_x)
                lum_re = box_mean(ii_lum, ey - er_y, ey + er_y, cx + d - er_x, cx + d + er_x)
                if abs(lum_le - lum_re) > 46.0:
                    continue

                lum_bridge = box_mean(ii_lum, ey - er_y, ey + er_y + 1, cx - max(2, d // 4), cx + max(2, d // 4))
                lum_fh = box_mean(ii_lum, ey - int(d * 1.0), ey - int(d * 0.42), cx - int(d * 0.65), cx + int(d * 0.65))
                lum_cheek_l = box_mean(ii_lum, ey + int(d * 0.35), ey + int(d * 0.95), cx - d - er_x, cx - max(2, d // 3))
                lum_cheek_r = box_mean(ii_lum, ey + int(d * 0.35), ey + int(d * 0.95), cx + max(2, d // 3), cx + d + er_x)

                # Both eyes must individually be darker than their own cheek and nose bridge
                left_c = min(lum_cheek_l - lum_le, lum_bridge - lum_le)
                right_c = min(lum_cheek_r - lum_re, lum_bridge - lum_re)
                if left_c <= 1.2 or right_c <= 1.2:
                    continue

                eye_dark_contrast = (
                    min(left_c, right_c) * 1.45
                    + (lum_fh - 0.5 * (lum_le + lum_re)) * 0.60
                    + 0.5 * (left_c + right_c) * 0.55
                )
                symmetry_pen = abs(lum_le - lum_re) * 1.15
                skin_conf = 0.25 + 0.55 * sk_fh + 0.55 * min(sk_eye_l, sk_eye_r) + 0.35 * sk_ch
                prior = 1.0 - 0.08 * abs((cx / float(sw)) - 0.5)
                score = (eye_dark_contrast - symmetry_pen) * skin_conf * prior

                if score > best_score:
                    best_score = score
                    best_cfg = (cx / float(sw), ey / float(sh), (2.0 * d) / float(sw))

    detected = best_cfg is not None and best_score > 3.5
    if detected and best_cfg is not None:
        center_x, eye_y_norm, ipd_w = best_cfg
    else:
        # Adaptive skin-centroid fallback (NEVER blind hardcoded 0.50/0.40 if skin region is visible)
        peak_prob = float(np.max(skin_s)) if skin_s.size else 0.0
        fg_mask = skin_s > max(0.18, peak_prob * 0.45)
        ys, xs = np.where(fg_mask)
        if ys.size > 24:
            x_lo_s, x_hi_s = float(np.percentile(xs, 10)), float(np.percentile(xs, 90))
            y_lo_s, y_hi_s = float(np.percentile(ys, 8)), float(np.percentile(ys, 88))
            center_x = _clamp(0.5 * (x_lo_s + x_hi_s) / float(sw), 0.16, 0.84)
            face_w_s = max(0.18, (x_hi_s - x_lo_s) / float(sw))
            face_h_s = max(0.24, (y_hi_s - y_lo_s) / float(sh))
            eye_y_norm = _clamp((y_lo_s / float(sh)) + 0.30 * face_h_s, 0.16, 0.70)
            ipd_w = _clamp(face_w_s * 0.43, 0.14, 0.48)
            detected = True
        else:
            center_x = 0.50
            eye_y_norm = 0.40
            ipd_w = 0.25

    # Convert horizontal IPD (normalized to image width) to vertical proportion using aspect ratio
    ipd_px = ipd_w * float(w)
    half_ipd_px = ipd_px * 0.5

    # Refine individual left/right eye pupil centers slightly downward-biased to avoid dark eyebrows
    ey_px = int(round(eye_y_norm * h)) + max(1, int(round(ipd_px * 0.05)))
    lx_px = int(round((center_x - ipd_w * 0.5) * w))
    rx_px = int(round((center_x + ipd_w * 0.5) * w))
    v_up = max(2, int(round(ipd_px * 0.06)))
    v_dn = max(4, int(round(ipd_px * 0.16)))
    h_rad = max(3, int(round(ipd_px * 0.14)))

    left_eye_y_px = ey_px
    right_eye_y_px = ey_px
    if detected:
        l_y0, l_y1 = max(0, ey_px - v_up), min(h, ey_px + v_dn)
        l_x0, l_x1 = max(0, lx_px - h_rad // 2), min(w, lx_px + h_rad // 2)
        l_win = lum[l_y0:l_y1, l_x0:l_x1]
        if l_win.size > 0:
            left_eye_y_px = l_y0 + int(np.argmin(np.mean(l_win, axis=1)))
        r_y0, r_y1 = max(0, ey_px - v_up), min(h, ey_px + v_dn)
        r_x0, r_x1 = max(0, rx_px - h_rad // 2), min(w, rx_px + h_rad // 2)
        r_win = lum[r_y0:r_y1, r_x0:r_x1]
        if r_win.size > 0:
            right_eye_y_px = r_y0 + int(np.argmin(np.mean(r_win, axis=1)))

    left_eye_x = _clamp(lx_px / float(w), 0.08, 0.86)
    right_eye_x = _clamp(rx_px / float(w), left_eye_x + 0.08, 0.92)
    left_eye_y = _clamp(left_eye_y_px / float(h), 0.12, 0.72)
    right_eye_y = _clamp(right_eye_y_px / float(h), 0.12, 0.72)
    eye_y_norm = (left_eye_y + right_eye_y) * 0.5
    eye_y_px = eye_y_norm * float(h)

    dy = (right_eye_y - left_eye_y) * h
    dx = max(1.0, (right_eye_x - left_eye_x) * w)
    tilt_deg = _clamp(math.degrees(math.atan2(dy, dx)), -14.0, 14.0)

    # Estimate true face vertical/horizontal bounds around detected eye line
    # Look outward laterally along cheek band & downward along chin axis for skin boundary
    cx_px = int(round(center_x * w))
    cheek_y0 = max(0, int(round(eye_y_px + 0.25 * ipd_px)))
    cheek_y1 = min(h, int(round(eye_y_px + 1.10 * ipd_px)))
    half_face_px = 1.25 * ipd_px
    if detected and cheek_y1 > cheek_y0 + 4:
        # Measure skin width across cheek strip around center_x
        cheek_strip = np.mean(skin_prob[cheek_y0:cheek_y1, :], axis=0)
        # Step outward left/right from cx_px up to max 2.0*ipd_px until skin prob drops
        max_lat = min(int(round(2.0 * ipd_px)), w // 2)
        l_span = int(round(1.1 * ipd_px))
        for off in range(int(round(0.6 * ipd_px)), max_lat):
            if cx_px - off <= 2 or cheek_strip[cx_px - off] < 0.18:
                l_span = off
                break
            l_span = off
        r_span = int(round(1.1 * ipd_px))
        for off in range(int(round(0.6 * ipd_px)), max_lat):
            if cx_px + off >= w - 3 or cheek_strip[cx_px + off] < 0.18:
                r_span = off
                break
            r_span = off
        measured_half = 0.5 * (l_span + r_span)
        half_face_px = _clamp(measured_half, 1.04 * ipd_px, 1.38 * ipd_px)

    half_face_w = _clamp(half_face_px / float(w), 0.16, 0.36)

    # Vertical canon tied to BOTH IPD & measured cheek width (oval head height ~ 2.45 * half_face_px)
    eff_scale = max(ipd_px, half_face_px * 0.85)
    y_top_px = max(4.0, eye_y_px - 0.92 * eff_scale)

    # Probe chin tip below lower lip shadow up to neck fold (~1.68..2.02 * eff_scale)
    chin_y_px = min(float(h - 6), eye_y_px + 1.88 * eff_scale)
    if detected:
        scan_lo = min(h - 10, int(round(eye_y_px + 1.62 * eff_scale)))
        scan_hi = min(h - 4, int(round(eye_y_px + 2.08 * eff_scale)))
        x_lo = max(0, int(round(cx_px - half_face_px * 0.30)))
        x_hi = min(w, int(round(cx_px + half_face_px * 0.30)))
        if scan_hi > scan_lo + 4 and x_hi > x_lo + 4:
            # Find row with darkest jaw-under-chin shadow weighted away from upper lip band
            chin_lum = np.mean(lum[scan_lo:scan_hi, x_lo:x_hi], axis=1)
            chin_skin = np.mean(skin_prob[scan_lo:scan_hi, x_lo:x_hi], axis=1)
            edge_score = (255.0 - chin_lum) * (0.6 + (1.0 - chin_skin))
            chin_y_px = float(scan_lo + int(np.argmax(edge_score)))
            chin_y_px = _clamp(
                chin_y_px,
                eye_y_px + 1.68 * eff_scale,
                min(float(h - 6), eye_y_px + 2.02 * eff_scale),
            )

    nose_y_px = _clamp(eye_y_px + 0.62 * eff_scale, eye_y_px + 10.0, chin_y_px - 18.0)

    # Refine mouth Y using Cr redness peak between nose tip and chin
    m_search_y0 = max(int(eye_y_px + 0.88 * eff_scale), int(nose_y_px + 6))
    m_search_y1 = min(int(chin_y_px - 10), int(eye_y_px + 1.38 * eff_scale))
    m_x0 = max(0, int(cx_px - 0.35 * eff_scale))
    m_x1 = min(w, int(cx_px + 0.35 * eff_scale))
    mouth_y_px = eye_y_px + 1.15 * eff_scale
    if detected and m_search_y1 > m_search_y0 + 3 and m_x1 > m_x0 + 3:
        lip_strip = np.mean(cr[m_search_y0:m_search_y1, m_x0:m_x1], axis=1)
        mouth_y_px = float(m_search_y0 + int(np.argmax(lip_strip)))
    mouth_y_px = _clamp(mouth_y_px, nose_y_px + 0.22 * eff_scale, chin_y_px - 0.28 * eff_scale)
    bx = _clamp(center_x - half_face_w, 0.04, 0.85)
    bw = _clamp(half_face_w * 2.0, 0.24, 0.84)
    by = _clamp(y_top_px / float(h), 0.03, 0.70)
    bh = _clamp((chin_y_px - y_top_px) / float(h), 0.24, 0.84)

    nose_y = _clamp(nose_y_px / float(h), eye_y_norm + 0.05, 0.85)
    mouth_y = _clamp(mouth_y_px / float(h), nose_y + 0.04, 0.90)
    chin_y = _clamp(chin_y_px / float(h), mouth_y + 0.04, 0.96)
    forehead_top_y = by
    face_h = max(0.20, chin_y - forehead_top_y)

    def pt(px: float, py: float) -> List[float]:
        return [round(_clamp(px, 0.02, 0.98), 4), round(_clamp(py, 0.02, 0.98), 4)]

    # Tight anatomical face oval locked strictly to forehead, temples, cheekbones, jaw, and chin
    contour: List[List[float]] = [
        pt(center_x, forehead_top_y),                                 # 0 Top forehead crown
        pt(center_x + half_face_w * 0.62, forehead_top_y + face_h * 0.06),  # 1 Right forehead
        pt(center_x + half_face_w * 0.96, eye_y_norm - face_h * 0.05),      # 2 Right temple
        pt(center_x + half_face_w * 1.00, eye_y_norm + face_h * 0.16),      # 3 Right cheekbone
        pt(center_x + half_face_w * 0.84, mouth_y - face_h * 0.02),         # 4 Right upper jaw
        pt(center_x + half_face_w * 0.56, mouth_y + face_h * 0.15),         # 5 Right lower jaw
        pt(center_x, chin_y),                                               # 6 Chin apex
        pt(center_x - half_face_w * 0.56, mouth_y + face_h * 0.15),         # 7 Left lower jaw
        pt(center_x - half_face_w * 0.84, mouth_y - face_h * 0.02),         # 8 Left upper jaw
        pt(center_x - half_face_w * 1.00, eye_y_norm + face_h * 0.16),      # 9 Left cheekbone
        pt(center_x - half_face_w * 0.96, eye_y_norm - face_h * 0.05),      # 10 Left temple
        pt(center_x - half_face_w * 0.62, forehead_top_y + face_h * 0.06),  # 11 Left forehead
    ]

    # T-Zone strictly spanning forehead + nasal bridge + tip
    t_zone: List[List[float]] = [
        pt(center_x - half_face_w * 0.58, forehead_top_y + face_h * 0.09),
        pt(center_x + half_face_w * 0.58, forehead_top_y + face_h * 0.09),
        pt(center_x + half_face_w * 0.52, eye_y_norm - face_h * 0.07),
        pt(center_x + half_face_w * 0.16, eye_y_norm + face_h * 0.02),
        pt(center_x + half_face_w * 0.22, nose_y + face_h * 0.03),
        pt(center_x, nose_y + face_h * 0.06),
        pt(center_x - half_face_w * 0.22, nose_y + face_h * 0.03),
        pt(center_x - half_face_w * 0.16, eye_y_norm + face_h * 0.02),
        pt(center_x - half_face_w * 0.52, eye_y_norm - face_h * 0.07),
    ]

    # U-Zone cheeks strictly under the eyes, inside the cheek contour
    left_cheek_zone: List[List[float]] = [
        pt(center_x - half_face_w * 0.86, eye_y_norm + face_h * 0.09),
        pt(center_x - half_face_w * 0.28, eye_y_norm + face_h * 0.10),
        pt(center_x - half_face_w * 0.26, mouth_y - face_h * 0.02),
        pt(center_x - half_face_w * 0.74, mouth_y - face_h * 0.01),
    ]
    right_cheek_zone: List[List[float]] = [
        pt(center_x + half_face_w * 0.28, eye_y_norm + face_h * 0.10),
        pt(center_x + half_face_w * 0.86, eye_y_norm + face_h * 0.09),
        pt(center_x + half_face_w * 0.74, mouth_y - face_h * 0.01),
        pt(center_x + half_face_w * 0.26, mouth_y - face_h * 0.02),
    ]

    return {
        "detected": detected,
        "tilt_deg": round(tilt_deg, 2),
        "bbox": {
            "x": round(bx, 4),
            "y": round(by, 4),
            "width": round(bw, 4),
            "height": round(bh, 4),
        },
        "landmarks": {
            "forehead": pt(center_x, forehead_top_y + face_h * 0.17),
            "left_eye": pt(left_eye_x, left_eye_y),
            "right_eye": pt(right_eye_x, right_eye_y),
            "nose_tip": pt(center_x, nose_y),
            "left_cheek": pt(center_x - half_face_w * 0.56, nose_y + face_h * 0.02),
            "right_cheek": pt(center_x + half_face_w * 0.56, nose_y + face_h * 0.02),
            "mouth_center": pt(center_x, mouth_y),
            "chin": pt(center_x, chin_y - face_h * 0.03),
        },
        "contour": contour,
        "t_zone": t_zone,
        "left_cheek_zone": left_cheek_zone,
        "right_cheek_zone": right_cheek_zone,
    }


def _build_face_oval_mask(w: int, h: int, geom: Dict[str, Any], blur_sigma: float = 0.0) -> np.ndarray:
    """Renders a crisp or soft-feathered float [0..1] facial mask inside geom['contour']."""
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    contour_px = [(int(round(p[0] * w)), int(round(p[1] * h))) for p in geom["contour"]]
    draw.polygon(contour_px, fill=255)
    if blur_sigma > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    return np.array(mask_img, dtype=np.float32) / 255.0


def _generate_woodlamp_image(norm_rgb: np.ndarray, geom: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Simulates continuous dermatological 365nm UV-A Wood's Lamp illumination
    anchored cleanly to the fitted face contour:
    - Smooth facial fluorescence without harsh binary blotches
    - Deep cobalt-indigo ambient vignette outside the face
    - Cyan-aqua sebum/pore fluorescence & melanin UV absorption contrast
    """
    r = norm_rgb[:, :, 0]
    g = norm_rgb[:, :, 1]
    b = norm_rgb[:, :, 2]
    h, w = r.shape

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    face_soft = _build_face_oval_mask(w, h, geom, blur_sigma=max(6.0, w * 0.045))
    face_bool = face_soft > 0.35

    # Local subsurface texture & pore fluorescence under UV
    local_base = _gaussian_blur_2d(lum, sigma=6.5)
    micro_scatter = lum - local_base
    sebum_glow = np.clip(micro_scatter / 15.0, 0.0, 1.0)
    melanin_abs = np.clip(1.0 - (lum / 225.0), 0.05, 0.88)
    erythema_norm = np.clip((cr - 135.0) / 36.0, 0.0, 1.0)

    # Normalized luminance preserved cleanly so facial features remain unmistakable
    lum_n = np.clip(lum / 255.0, 0.0, 1.0)

    # Inside face: true clinical Wood's UV cobalt + cyan sebum glow
    face_r = 16.0 + 36.0 * lum_n + 42.0 * sebum_glow + 38.0 * erythema_norm * lum_n
    face_g = 48.0 + 112.0 * lum_n + 78.0 * sebum_glow
    face_b = 126.0 + 122.0 * lum_n + 36.0 * sebum_glow

    # Outside face: muted dark UV examination room blue
    room_r = 10.0 + 18.0 * lum_n
    room_g = 22.0 + 42.0 * lum_n
    room_b = 58.0 + 88.0 * lum_n

    w_r = face_soft * face_r + (1.0 - face_soft) * room_r
    w_g = face_soft * face_g + (1.0 - face_soft) * room_g
    w_b = face_soft * face_b + (1.0 - face_soft) * room_b

    wood_arr = np.clip(np.stack([w_r, w_g, w_b], axis=2), 0.0, 255.0).astype(np.uint8)

    valid = face_bool if np.any(face_bool) else np.ones(lum.shape, dtype=bool)
    sebum_idx = float(np.clip(np.mean(sebum_glow[valid]) * 185.0 + 18.0, 5.0, 96.0))
    erythema_idx = float(np.clip(np.mean(erythema_norm[valid]) * 145.0, 5.0, 96.0))
    melanin_contrast = float(np.clip(np.std(melanin_abs[valid]) * 310.0, 8.0, 95.0))
    uniformity = float(np.clip(100.0 - np.std(lum[valid]) * 1.15, 18.0, 96.0))

    metrics = {
        "tzone_sebum_fluorescence": round(sebum_idx, 1),
        "erythema_index": round(erythema_idx, 1),
        "melanin_uv_contrast": round(melanin_contrast, 1),
        "uniformity_score": round(uniformity, 1),
    }
    return Image.fromarray(wood_arr, mode="RGB"), metrics


def _generate_geometry_mesh_image(norm_rgb: np.ndarray, geom: Dict[str, Any]) -> Image.Image:
    """
    Renders a crisp, dense clinical Blue & White anatomical wireframe fitted directly
    to the detected eyes, nose, cheeks, jawline, and T/U-zones.
    """
    arr = norm_rgb.astype(np.float32)
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])[:, :, None]
    cool_tint = 0.70 * arr + 0.30 * (lum * np.array([[[0.90, 0.95, 1.03]]], dtype=np.float32))
    base = Image.fromarray(np.clip(cool_tint, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def to_px(pt: List[float]) -> Tuple[int, int]:
        return (int(round(pt[0] * w)), int(round(pt[1] * h)))

    def mid(a: Tuple[int, int], b: Tuple[int, int], t: float = 0.5) -> Tuple[int, int]:
        return (int(round(a[0] * (1.0 - t) + b[0] * t)), int(round(a[1] * (1.0 - t) + b[1] * t)))

    c_px = [to_px(p) for p in geom["contour"]]
    tzone_px = [to_px(p) for p in geom["t_zone"]]
    lcheek_px = [to_px(p) for p in geom["left_cheek_zone"]]
    rcheek_px = [to_px(p) for p in geom["right_cheek_zone"]]

    # Clean soft blue zone tints
    draw.polygon(tzone_px, fill=(30, 108, 239, 36), outline=(37, 99, 235, 210))
    draw.polygon(lcheek_px, fill=(14, 165, 233, 32), outline=(56, 189, 248, 200))
    draw.polygon(rcheek_px, fill=(14, 165, 233, 32), outline=(56, 189, 248, 200))

    lm = geom["landmarks"]
    fh = to_px(lm["forehead"])
    le = to_px(lm["left_eye"])
    re = to_px(lm["right_eye"])
    nt = to_px(lm["nose_tip"])
    lc = to_px(lm["left_cheek"])
    rc = to_px(lm["right_cheek"])
    mc = to_px(lm["mouth_center"])
    ch = to_px(lm["chin"])

    # Additional anatomical anchor midpoints (glabella, nose bridge, mouth corners)
    glab = mid(le, re, 0.5)
    n_bridge = mid(glab, nt, 0.5)
    l_lip = mid(mc, c_px[8], 0.36)
    r_lip = mid(mc, c_px[4], 0.36)
    l_brow = mid(le, c_px[11], 0.42)
    r_brow = mid(re, c_px[1], 0.42)

    mesh_edges = [
        # Upper cranium / brow arch
        (c_px[11], fh), (fh, c_px[1]),
        (c_px[10], l_brow), (l_brow, fh), (fh, r_brow), (r_brow, c_px[2]),
        (l_brow, glab), (r_brow, glab), (fh, glab),
        # Orbital & nasal bridge
        (c_px[10], le), (l_brow, le), (le, glab),
        (c_px[2], re), (r_brow, re), (re, glab),
        (glab, n_bridge), (n_bridge, nt), (le, n_bridge), (re, n_bridge),
        # Midface cheekbones & nasolabial
        (c_px[9], lc), (le, lc), (n_bridge, lc), (nt, lc),
        (c_px[3], rc), (re, rc), (n_bridge, rc), (nt, rc),
        # Perioral & jawline
        (nt, l_lip), (nt, r_lip), (nt, mc),
        (lc, l_lip), (rc, r_lip), (l_lip, mc), (r_lip, mc),
        (c_px[8], l_lip), (c_px[4], r_lip),
        (c_px[8], mc), (c_px[4], mc),
        (l_lip, c_px[7]), (r_lip, c_px[5]), (mc, ch),
        (c_px[7], ch), (c_px[5], ch), (lc, c_px[8]), (rc, c_px[4]),
    ]

    thin_w = max(1, int(round(min(w, h) * 0.004)))
    for a, b in mesh_edges:
        draw.line([a, b], fill=(219, 234, 254, 195), width=thin_w)

    # Outer fitted contour in crisp AURA Blue
    line_w = max(2, int(round(min(w, h) * 0.006)))
    for i in range(len(c_px)):
        p1 = c_px[i]
        p2 = c_px[(i + 1) % len(c_px)]
        draw.line([p1, p2], fill=(30, 108, 239, 235), width=line_w)

    r_outer = max(3, int(round(min(w, h) * 0.011)))
    r_inner = max(2, r_outer - 1)
    for key_pt in [fh, le, re, glab, nt, lc, rc, mc, ch]:
        draw.ellipse(
            [key_pt[0] - r_outer, key_pt[1] - r_outer, key_pt[0] + r_outer, key_pt[1] + r_outer],
            fill=(255, 255, 255, 245),
            outline=(30, 108, 239, 255),
            width=1,
        )
        draw.ellipse(
            [key_pt[0] - r_inner + 1, key_pt[1] - r_inner + 1, key_pt[0] + r_inner - 1, key_pt[1] + r_inner - 1],
            fill=(30, 108, 239, 255),
        )

    return Image.alpha_composite(base, overlay).convert("RGB")


def _generate_micro_analysis_image(
    norm_rgb: np.ndarray, skin_prob: np.ndarray, geom: Dict[str, Any]
) -> Image.Image:
    """
    Computes a clean, subtle ~384 micro-patch skin biomarker map clipped strictly
    inside the fitted facial contour while keeping the face clearly visible underneath.
    """
    r = norm_rgb[:, :, 0]
    g = norm_rgb[:, :, 1]
    b = norm_rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Keep portrait crisp with subtle cool clinical tone
    cool_base = np.stack([
        0.55 * r + 0.45 * lum * 0.94,
        0.55 * g + 0.45 * lum * 0.97,
        0.55 * b + 0.45 * lum * 1.03,
    ], axis=2)
    base_rgba = Image.fromarray(np.clip(cool_base, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    w, h = base_rgba.size

    face_poly = Image.new("L", (w, h), 0)
    poly_draw = ImageDraw.Draw(face_poly)
    contour_px = [(int(round(p[0] * w)), int(round(p[1] * h))) for p in geom["contour"]]
    poly_draw.polygon(contour_px, fill=255)
    face_arr = np.array(face_poly) > 128

    # Keep eye bands gentle so eyes remain natural and clear
    lm = geom["landmarks"]
    ley_y = int(round(lm["left_eye"][1] * h))
    rey_y = int(round(lm["right_eye"][1] * h))
    eye_half_r = max(6, int(round(geom["bbox"]["width"] * w * 0.10)))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = geom["bbox"]
    fx0 = max(0, min(w - 2, int(bbox["x"] * w)))
    fy0 = max(0, min(h - 2, int(bbox["y"] * h)))
    fx1 = max(fx0 + 1, min(w, int((bbox["x"] + bbox["width"]) * w)))
    fy1 = max(fy0 + 1, min(h, int((bbox["y"] + bbox["height"]) * h)))

    cols, rows = 18, 22
    cell_w = max(4.0, (fx1 - fx0) / float(cols))
    cell_h = max(4.0, (fy1 - fy0) / float(rows))

    face_cr = cr[fy0:fy1, fx0:fx1]
    baseline_cr = float(np.median(face_cr)) if face_cr.size > 0 else 140.0

    for row in range(rows):
        for col in range(cols):
            cx0 = max(0, min(w - 1, int(round(fx0 + col * cell_w))))
            cy0 = max(0, min(h - 1, int(round(fy0 + row * cell_h))))
            cx1 = max(cx0, min(w, int(round(fx0 + (col + 1) * cell_w))))
            cy1 = max(cy0, min(h, int(round(fy0 + (row + 1) * cell_h))))
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            patch_mask = face_arr[cy0:cy1, cx0:cx1]
            if patch_mask.size == 0 or np.mean(patch_mask) < 0.55:
                continue

            # Subtle transparency inside orbital region so pupils remain crisp
            pm_x = (cx0 + cx1) * 0.5
            pm_y = (cy0 + cy1) * 0.5
            near_eye = (
                math.hypot(pm_x - lm["left_eye"][0] * w, pm_y - ley_y) < eye_half_r
                or math.hypot(pm_x - lm["right_eye"][0] * w, pm_y - rey_y) < eye_half_r
            )
            alpha_scale = 0.38 if near_eye else 1.0

            patch_cr = float(np.mean(cr[cy0:cy1, cx0:cx1]))
            patch_std = float(np.std(lum[cy0:cy1, cx0:cx1]))
            stress = _clamp(((patch_cr - baseline_cr) / 14.0) * 0.65 + (patch_std / 24.0) * 0.35, 0.0, 1.0)

            if stress < 0.38:
                fill_col = (30, 108, 239, int(52 * alpha_scale))
            elif stress < 0.64:
                fill_col = (14, 165, 233, int(64 * alpha_scale))
            elif stress < 0.82:
                fill_col = (245, 158, 11, int(76 * alpha_scale))
            else:
                fill_col = (239, 68, 68, int(86 * alpha_scale))

            draw.rectangle([cx0, cy0, cx1 - 1, cy1 - 1], fill=fill_col, outline=(255, 255, 255, int(48 * alpha_scale)))

    # Subtle outer face boundary ring
    for i in range(len(contour_px)):
        p1 = contour_px[i]
        p2 = contour_px[(i + 1) % len(contour_px)]
        draw.line([p1, p2], fill=(255, 255, 255, 165), width=max(1, int(round(w * 0.004))))

    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def _crop_face_to_geometry(
    pil_img: Image.Image, geom: Dict[str, Any]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Crops image tightly to the fitted facial contour + subtle 6% margin so the
    resulting portrait contains only the tracked face (no room/clothing clutter)
    and re-normalizes all geometry coordinates into [0..1] of the cropped image.
    """
    contour = geom.get("contour") or []
    if not contour or not geom.get("detected"):
        return pil_img, geom

    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_w = max(0.20, max_x - min_x)
    span_h = max(0.24, max_y - min_y)

    # Skip re-crop if face already fills >78% width & >78% height
    if span_w > 0.78 and span_h > 0.78:
        return pil_img, geom

    pad_x = span_w * 0.065
    pad_y = span_h * 0.065
    cx0 = _clamp(min_x - pad_x, 0.0, 0.85)
    cy0 = _clamp(min_y - pad_y, 0.0, 0.85)
    cx1 = _clamp(max_x + pad_x, cx0 + 0.15, 1.0)
    cy1 = _clamp(max_y + pad_y, cy0 + 0.15, 1.0)

    ow, oh = pil_img.size
    px0 = max(0, int(round(cx0 * ow)))
    py0 = max(0, int(round(cy0 * oh)))
    px1 = min(ow, int(round(cx1 * ow)))
    py1 = min(oh, int(round(cy1 * oh)))
    if px1 - px0 < 32 or py1 - py0 < 32:
        return pil_img, geom

    cropped = pil_img.crop((px0, py0, px1, py1))
    cw = max(1e-5, (px1 - px0) / float(ow))
    ch = max(1e-5, (py1 - py0) / float(oh))
    nx0 = px0 / float(ow)
    ny0 = py0 / float(oh)

    def remap(pt: List[float]) -> List[float]:
        return [
            round(_clamp((pt[0] - nx0) / cw, 0.01, 0.99), 4),
            round(_clamp((pt[1] - ny0) / ch, 0.01, 0.99), 4),
        ]

    bb = geom["bbox"]
    new_geom = {
        "detected": geom["detected"],
        "tilt_deg": geom["tilt_deg"],
        "bbox": {
            "x": round(_clamp((bb["x"] - nx0) / cw, 0.01, 0.95), 4),
            "y": round(_clamp((bb["y"] - ny0) / ch, 0.01, 0.95), 4),
            "width": round(_clamp(bb["width"] / cw, 0.10, 0.98), 4),
            "height": round(_clamp(bb["height"] / ch, 0.10, 0.98), 4),
        },
        "landmarks": {k: remap(v) for k, v in geom["landmarks"].items()},
        "contour": [remap(p) for p in geom["contour"]],
        "t_zone": [remap(p) for p in geom["t_zone"]],
        "left_cheek_zone": [remap(p) for p in geom["left_cheek_zone"]],
        "right_cheek_zone": [remap(p) for p in geom["right_cheek_zone"]],
    }
    return cropped, new_geom


def _apply_face_studio_isolation(arr_rgb: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    """
    Softly dims non-facial background corners outside the fitted geometry contour
    so only the tracked face stands out across cropped photo and CV overlays.
    """
    h, w, _ = arr_rgb.shape
    mask = _build_face_oval_mask(w, h, geom, blur_sigma=3.5)[:, :, None]
    bg_studio = np.array([15.0, 32.0, 74.0], dtype=np.float32)[None, None, :]
    isolated = arr_rgb.astype(np.float32) * (0.14 + 0.86 * mask) + bg_studio * (1.0 - mask) * 0.76
    return np.clip(isolated, 0.0, 255.0)


def run_skin_cv_pipeline(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes the full Python Computer Vision skin pipeline in pure NumPy + Pillow:
      1. EXIF transpose correction so mobile selfies align upright
      2. YCbCr skin probability + Haar-like eye-pair facial geometry fitting
      3. Tight face-only crop & studio isolation to the detected geometry
      4. Smooth Wood's Lamp 365nm simulation, anatomical mesh & 384+ micro-patch map
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Step 1: Detect real face geometry on full upright photo
    detect_img = pil_img.copy()
    detect_img.thumbnail((420, 420), Image.Resampling.LANCZOS)
    detect_rgb = np.array(detect_img, dtype=np.float32)
    detect_skin = _compute_skin_probability(detect_rgb)
    raw_geom = _detect_face_geometry(detect_rgb, detect_skin)

    # Step 2: Crop image strictly to the fitted face geometry + re-normalize coords
    face_pil, geom = _crop_face_to_geometry(pil_img, raw_geom)
    crop_w, crop_h = face_pil.size

    work_img = face_pil.copy()
    work_img.thumbnail((480, 480), Image.Resampling.LANCZOS)
    rgb = np.array(work_img, dtype=np.uint8)

    norm_rgb = _normalize_lighting_and_dull_razor(rgb)
    skin_prob = _compute_skin_probability(norm_rgb)

    isolated_raw = _apply_face_studio_isolation(rgb.astype(np.float32), geom)
    isolated_norm = _apply_face_studio_isolation(norm_rgb, geom)

    wood_img, wood_metrics = _generate_woodlamp_image(norm_rgb, geom)
    mesh_img = _generate_geometry_mesh_image(isolated_norm, geom)
    micro_img = _generate_micro_analysis_image(isolated_norm, skin_prob, geom)
    clean_crop_img = Image.fromarray(np.clip(isolated_raw, 0, 255).astype(np.uint8), mode="RGB")

    face_mask = _build_face_oval_mask(work_img.width, work_img.height, geom) > 0.5
    skin_cov_pct = round(float(np.mean((skin_prob > 0.25) & face_mask) * 100.0 / max(0.05, float(np.mean(face_mask)))), 1)
    skin_cov_pct = min(100.0, skin_cov_pct)

    cv_metrics = {
        **wood_metrics,
        "skin_coverage_pct": skin_cov_pct,
        "image_width": crop_w,
        "image_height": crop_h,
    }

    findings: List[str] = []
    if wood_metrics["tzone_sebum_fluorescence"] >= 58:
        findings.append(
            f"Fluoresensi Wood's Lamp menunjukkan akumulasi sebum T-zone tinggi ({wood_metrics['tzone_sebum_fluorescence']}/100)"
        )
    else:
        findings.append(
            f"Fluoresensi lipid Wood's Lamp terkendali ({wood_metrics['tzone_sebum_fluorescence']}/100)"
        )
    if wood_metrics["erythema_index"] >= 50:
        findings.append(f"Indeks eritema / kemerahan vaskular meningkat ({wood_metrics['erythema_index']}/100)")
    if wood_metrics["melanin_uv_contrast"] >= 45:
        findings.append(f"Kontras penyerapan melanin UV cukup signifikan ({wood_metrics['melanin_uv_contrast']}/100)")

    return {
        "face_geometry": geom,
        "cropped_face_image": _encode_jpeg_data_url(clean_crop_img, max_width=420, quality=86),
        "woodlamp_image": _encode_jpeg_data_url(wood_img, max_width=420, quality=84),
        "geometry_mesh_image": _encode_jpeg_data_url(mesh_img, max_width=420, quality=85),
        "micro_analysis_image": _encode_jpeg_data_url(micro_img, max_width=420, quality=85),
        "cv_metrics": cv_metrics,
        "woods_lamp_findings": findings,
    }
