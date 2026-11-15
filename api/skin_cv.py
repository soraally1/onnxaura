import base64
import io
import math
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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
    """Morphological grayscale closing (dilation followed by erosion) via Pillow C filters."""
    u8 = np.clip(arr2d, 0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    closed = img.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.array(closed, dtype=np.float32)


def _morph_clean_mask(mask: np.ndarray) -> np.ndarray:
    """Applies binary morphological opening -> closing via Pillow C filters."""
    u8 = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(u8, mode="L")
    opened = img.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    closed = opened.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    return np.array(closed, dtype=np.uint8) > 127


def _largest_component_and_fill_holes(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Finds the largest centrally weighted connected component in pure Python/NumPy
    on a compact downsampled grid (<=96px) for sub-5ms serverless performance.
    """
    h, w = mask.shape
    step = max(1, min(h, w) // 72)
    small = mask[::step, ::step]
    sh, sw = small.shape

    visited = np.zeros((sh, sw), dtype=bool)
    best_pixels: List[Tuple[int, int]] = []
    best_score = -1.0
    img_cx, img_cy = sw * 0.5, sh * 0.46
    min_area = max(12, int(sh * sw * 0.035))

    for sy in range(sh):
        for sx in range(sw):
            if not small[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            comp: List[Tuple[int, int]] = []
            sum_y = 0
            sum_x = 0
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                sum_y += cy
                sum_x += cx
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            area = len(comp)
            if area < min_area:
                continue
            mean_y = sum_y / float(area)
            mean_x = sum_x / float(area)
            dist_penalty = math.hypot((mean_x - img_cx) / max(1, sw), (mean_y - img_cy) / max(1, sh))
            score = area * max(0.35, 1.0 - 0.75 * dist_penalty)
            if score > best_score:
                best_score = score
                best_pixels = comp

    if not best_pixels:
        return np.zeros((h, w), dtype=bool), False

    comp_small = np.zeros((sh, sw), dtype=np.uint8)
    for cy, cx in best_pixels:
        comp_small[cy, cx] = 255

    # Fill internal holes inside component using border flood fill on small mask
    outside = np.zeros((sh, sw), dtype=bool)
    q_bg: deque = deque()
    for bx in range(sw):
        if comp_small[0, bx] == 0:
            outside[0, bx] = True
            q_bg.append((0, bx))
        if comp_small[sh - 1, bx] == 0 and not outside[sh - 1, bx]:
            outside[sh - 1, bx] = True
            q_bg.append((sh - 1, bx))
    for by in range(sh):
        if comp_small[by, 0] == 0 and not outside[by, 0]:
            outside[by, 0] = True
            q_bg.append((by, 0))
        if comp_small[by, sw - 1] == 0 and not outside[by, sw - 1]:
            outside[by, sw - 1] = True
            q_bg.append((by, sw - 1))

    while q_bg:
        cy, cx = q_bg.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < sh and 0 <= nx < sw and comp_small[ny, nx] == 0 and not outside[ny, nx]:
                outside[ny, nx] = True
                q_bg.append((ny, nx))

    filled_small = (~outside).astype(np.uint8) * 255
    upscaled = Image.fromarray(filled_small, mode="L").resize((w, h), Image.Resampling.BILINEAR)
    return (np.array(upscaled, dtype=np.uint8) > 127), True


def _encode_jpeg_data_url(image: Image.Image, max_width: int = 380, quality: int = 80) -> str:
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
    scales = np.clip(gray_target / means, 0.78, 1.28)
    balanced = np.clip(arr * scales[None, None, :], 0.0, 255.0)
    constancy = 0.55 * arr + 0.45 * balanced

    # Dull Razor morphological black-hat on luminance for stray hairs & specular glare
    lum = 0.299 * constancy[:, :, 0] + 0.587 * constancy[:, :, 1] + 0.114 * constancy[:, :, 2]
    closed_lum = _grey_closing_2d(lum, size=5)
    hair_response = closed_lum - lum

    sat = (np.max(constancy, axis=2) - np.min(constancy, axis=2)) / (np.max(constancy, axis=2) + 1e-5)
    artifact_mask = (hair_response > 15.0) | ((lum > 240.0) & (sat < 0.14))
    artifact_weight = _gaussian_blur_2d(artifact_mask.astype(np.float32), sigma=1.1)
    artifact_weight = np.clip(artifact_weight * 1.35, 0.0, 0.85)[:, :, None]

    smoothed = _gaussian_blur_rgb(constancy, sigma=1.5)
    cleaned = constancy * (1.0 - artifact_weight) + smoothed * artifact_weight
    return np.clip(cleaned, 0.0, 255.0)


def _segment_skin_ycbcr_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Segments human facial skin pixels across all Fitzpatrick phototypes
    using BT.601 YCbCr bounds paired with warm chromatic dominance.
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y_lum = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    ycbcr_skin = (cb >= 76.0) & (cb <= 126.0) & (cr >= 134.0) & (cr <= 174.0) & (y_lum > 35.0)
    deep_tone_skin = (
        (r > g * 1.05)
        & (r > b * 1.14)
        & ((r - np.minimum(g, b)) > 14.0)
        & (cr >= 134.0)
        & (cr <= 178.0)
        & (cb >= 74.0)
        & (cb <= 126.0)
        & (y_lum > 28.0)
    )
    mask = ycbcr_skin | deep_tone_skin
    return _morph_clean_mask(mask)


def _smooth_1d(arr: np.ndarray) -> np.ndarray:
    """Lightweight 1D box/binomial smoother replacing gaussian_filter on 1D strips."""
    if arr.size < 3:
        return arr
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _detect_face_geometry(rgb: np.ndarray, skin_mask: np.ndarray) -> Dict[str, Any]:
    """
    Automatically localizes the user's face bounding box and fits anatomical
    facial geometry (brow/eye line, nose bridge, cheek contours, mouth, chin)
    purely via NumPy + Pillow.
    """
    h, w, _ = rgb.shape
    best_mask, detected = _largest_component_and_fill_holes(skin_mask)

    if not detected:
        x0, x1 = int(w * 0.24), int(w * 0.76)
        y0, y1 = int(h * 0.14), int(h * 0.80)
        center_x = 0.50
    else:
        ys, xs = np.where(best_mask)
        y0 = int(np.percentile(ys, 2))
        y1_raw = int(np.percentile(ys, 98))
        x0 = int(np.percentile(xs, 3))
        x1 = int(np.percentile(xs, 97))

        box_w = max(20, x1 - x0)
        max_face_h = int(box_w * 1.42)
        if (y1_raw - y0) > max_face_h:
            scan_start = y0 + int(max_face_h * 0.78)
            scan_end = min(y1_raw, y0 + int(max_face_h * 1.15))
            if scan_end > scan_start:
                row_widths = np.sum(best_mask[scan_start:scan_end, x0:x1], axis=1)
                mid_cheek_slice = best_mask[y0 + int(max_face_h * 0.35):scan_start, x0:x1]
                mid_cheek_w = (
                    float(np.percentile(np.sum(mid_cheek_slice, axis=1), 75))
                    if mid_cheek_slice.size > 0
                    else float(box_w * 0.7)
                )
                narrow_rows = np.where(row_widths < max(12.0, mid_cheek_w * 0.56))[0]
                y1 = (scan_start + int(narrow_rows[0])) if len(narrow_rows) > 0 else (y0 + max_face_h)
            else:
                y1 = y0 + max_face_h
        else:
            y1 = y1_raw

        upper_cut = y0 + max(10, int((y1 - y0) * 0.72))
        _, upper_xs = np.where(best_mask[y0:upper_cut, x0:x1])
        center_x = ((x0 + float(np.mean(upper_xs))) / float(w)) if len(upper_xs) > 0 else (((x0 + x1) * 0.5) / float(w))

    bx = _clamp(x0 / float(w), 0.05, 0.70)
    by = _clamp(y0 / float(h), 0.04, 0.70)
    bw = _clamp((x1 - x0) / float(w), 0.24, 0.88)
    bh = _clamp((y1 - y0) / float(h), 0.30, 0.88)
    center_x = _clamp(center_x, bx + bw * 0.25, bx + bw * 0.75)

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    cr = 128.0 + 0.5 * rgb[:, :, 0] - 0.418688 * rgb[:, :, 1] - 0.081312 * rgb[:, :, 2]

    eye_band_top = int(h * (by + bh * 0.24))
    eye_band_bot = int(h * (by + bh * 0.48))
    mid_x0 = max(0, int(w * (center_x - bw * 0.28)))
    mid_x1 = min(w, int(w * (center_x + bw * 0.28)))

    eye_y_rel = 0.36
    left_eye_x = _clamp(center_x - bw * 0.19, bx + bw * 0.14, center_x - bw * 0.08)
    right_eye_x = _clamp(center_x + bw * 0.19, center_x + bw * 0.08, bx + bw * 0.86)
    left_eye_y = by + bh * eye_y_rel
    right_eye_y = by + bh * eye_y_rel

    if detected and eye_band_bot > eye_band_top + 4 and mid_x1 > mid_x0 + 8:
        eye_prof = _smooth_1d(np.mean(lum[eye_band_top:eye_band_bot, mid_x0:mid_x1], axis=1))
        eye_px_y = eye_band_top + int(np.argmin(eye_prof))

        row_lo = max(0, eye_px_y - max(3, int(h * 0.025)))
        row_hi = min(h, eye_px_y + max(3, int(h * 0.025)))
        cx_px = int(w * center_x)

        orbit_half = min(bw * 0.27, 0.22)
        l_start = max(0, int(w * (center_x - orbit_half)))
        l_end = max(l_start + 4, cx_px - max(4, int(w * bw * 0.06)))
        if l_end > l_start:
            l_strip = _smooth_1d(np.mean(lum[row_lo:row_hi, l_start:l_end], axis=0))
            left_eye_x = (l_start + int(np.argmin(l_strip))) / float(w)
            l_col = int(left_eye_x * w)
            l_vstrip = lum[eye_band_top:eye_band_bot, max(0, l_col - 3):min(w, l_col + 4)]
            if l_vstrip.size > 0:
                left_eye_y = (eye_band_top + int(np.argmin(np.mean(l_vstrip, axis=1)))) / float(h)

        r_start = min(w - 4, cx_px + max(4, int(w * bw * 0.06)))
        r_end = min(w, int(w * (center_x + orbit_half)))
        if r_end > r_start:
            r_strip = _smooth_1d(np.mean(lum[row_lo:row_hi, r_start:r_end], axis=0))
            right_eye_x = (r_start + int(np.argmin(r_strip))) / float(w)
            r_col = int(right_eye_x * w)
            r_vstrip = lum[eye_band_top:eye_band_bot, max(0, r_col - 3):min(w, r_col + 4)]
            if r_vstrip.size > 0:
                right_eye_y = (eye_band_top + int(np.argmin(np.mean(r_vstrip, axis=1)))) / float(h)

    eye_y_norm = _clamp((left_eye_y + right_eye_y) * 0.5, by + bh * 0.26, by + bh * 0.46)
    dy = (right_eye_y - left_eye_y) * h
    dx = max(1e-3, (right_eye_x - left_eye_x) * w)
    tilt_deg = _clamp(math.degrees(math.atan2(dy, dx)), -14.0, 14.0)

    mouth_y = by + bh * 0.74
    m_top = int(h * (by + bh * 0.63))
    m_bot = int(h * (by + bh * 0.86))
    m_x0 = max(0, int(w * (center_x - bw * 0.18)))
    m_x1 = min(w, int(w * (center_x + bw * 0.18)))
    if detected and m_bot > m_top + 4 and m_x1 > m_x0 + 4:
        lip_prof = _smooth_1d(np.mean(cr[m_top:m_bot, m_x0:m_x1], axis=1))
        mouth_px_y = m_top + int(np.argmax(lip_prof))
        mouth_y = _clamp(mouth_px_y / float(h), eye_y_norm + bh * 0.22, by + bh * 0.87)

    nose_y = _clamp(eye_y_norm + (mouth_y - eye_y_norm) * 0.54, eye_y_norm + 0.04, mouth_y - 0.03)
    nose_x = center_x

    def pt(px: float, py: float) -> List[float]:
        return [round(_clamp(px, 0.02, 0.98), 4), round(_clamp(py, 0.02, 0.98), 4)]

    contour: List[List[float]] = [
        pt(center_x, by + bh * 0.02),
        pt(center_x + bw * 0.28, by + bh * 0.08),
        pt(bx + bw * 0.96, eye_y_norm - bh * 0.04),
        pt(bx + bw * 0.95, nose_y),
        pt(bx + bw * 0.84, mouth_y + bh * 0.02),
        pt(center_x + bw * 0.22, by + bh * 0.94),
        pt(center_x, by + bh * 0.98),
        pt(center_x - bw * 0.22, by + bh * 0.94),
        pt(bx + bw * 0.16, mouth_y + bh * 0.02),
        pt(bx + bw * 0.05, nose_y),
        pt(bx + bw * 0.04, eye_y_norm - bh * 0.04),
        pt(center_x - bw * 0.28, by + bh * 0.08),
    ]

    t_zone: List[List[float]] = [
        pt(center_x - bw * 0.26, by + bh * 0.13),
        pt(center_x + bw * 0.26, by + bh * 0.13),
        pt(center_x + bw * 0.23, eye_y_norm - bh * 0.06),
        pt(center_x + bw * 0.08, eye_y_norm + bh * 0.02),
        pt(center_x + bw * 0.11, nose_y + bh * 0.03),
        pt(center_x, nose_y + bh * 0.06),
        pt(center_x - bw * 0.11, nose_y + bh * 0.03),
        pt(center_x - bw * 0.08, eye_y_norm + bh * 0.02),
        pt(center_x - bw * 0.23, eye_y_norm - bh * 0.06),
    ]

    left_cheek_zone: List[List[float]] = [
        pt(bx + bw * 0.11, eye_y_norm + bh * 0.08),
        pt(center_x - bw * 0.15, eye_y_norm + bh * 0.09),
        pt(center_x - bw * 0.15, mouth_y + bh * 0.04),
        pt(bx + bw * 0.17, mouth_y + bh * 0.03),
    ]
    right_cheek_zone: List[List[float]] = [
        pt(center_x + bw * 0.15, eye_y_norm + bh * 0.09),
        pt(bx + bw * 0.89, eye_y_norm + bh * 0.08),
        pt(bx + bw * 0.83, mouth_y + bh * 0.03),
        pt(center_x + bw * 0.15, mouth_y + bh * 0.04),
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
            "forehead": pt(center_x, by + bh * 0.17),
            "left_eye": pt(left_eye_x, left_eye_y),
            "right_eye": pt(right_eye_x, right_eye_y),
            "nose_tip": pt(nose_x, nose_y),
            "left_cheek": pt(center_x - bw * 0.26, nose_y + bh * 0.04),
            "right_cheek": pt(center_x + bw * 0.26, nose_y + bh * 0.04),
            "mouth_center": pt(center_x, mouth_y),
            "chin": pt(center_x, by + bh * 0.94),
        },
        "contour": contour,
        "t_zone": t_zone,
        "left_cheek_zone": left_cheek_zone,
        "right_cheek_zone": right_cheek_zone,
    }


def _generate_woodlamp_image(norm_rgb: np.ndarray, skin_mask: np.ndarray) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Simulates dermatological 365nm UV-A Wood's Lamp illumination from RGB face data.
    """
    r = norm_rgb[:, :, 0]
    g = norm_rgb[:, :, 1]
    b = norm_rgb[:, :, 2]

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin_f = _gaussian_blur_2d(skin_mask.astype(np.float32), sigma=2.0)
    bg_f = 1.0 - skin_f

    local_base = _gaussian_blur_2d(lum, sigma=7.0)
    micro_scatter = lum - local_base
    pore_dip = np.clip(-micro_scatter / 18.0, 0.0, 1.0)
    sebum_glow = np.clip(micro_scatter / 16.0, 0.0, 1.0)

    melanin_abs = np.clip(1.0 - (lum / 220.0), 0.05, 0.90)
    erythema_norm = np.clip((cr - 134.0) / 38.0, 0.0, 1.0)

    bg_lum = lum * 0.22
    w_r = (
        skin_f * (12.0 + 28.0 * (1.0 - melanin_abs) + 92.0 * erythema_norm * (0.45 + 0.55 * pore_dip) + 46.0 * sebum_glow)
        + bg_f * (8.0 + bg_lum * 0.25)
    )
    w_g = (
        skin_f * (38.0 + 118.0 * (0.72 - 0.42 * melanin_abs) + 102.0 * sebum_glow)
        + bg_f * (18.0 + bg_lum * 0.40)
    )
    w_b = (
        skin_f * (108.0 + 136.0 * (0.88 - 0.30 * melanin_abs) + 52.0 * sebum_glow)
        + bg_f * (52.0 + bg_lum * 0.72)
    )

    wood_arr = np.clip(np.stack([w_r, w_g, w_b], axis=2), 0.0, 255.0).astype(np.uint8)

    valid = skin_mask if np.any(skin_mask) else np.ones(lum.shape, dtype=bool)
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
    Renders clean Blue & White clinical anatomical wireframe, T-Zone / U-Zone boundaries,
    and fitted facial landmarks directly onto the user's face proportions.
    """
    arr = norm_rgb.astype(np.float32)
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])[:, :, None]
    cool_tint = 0.62 * arr + 0.38 * (lum * np.array([[[0.88, 0.94, 1.04]]], dtype=np.float32))
    base = Image.fromarray(np.clip(cool_tint, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def to_px(pt: List[float]) -> Tuple[int, int]:
        return (int(round(pt[0] * w)), int(round(pt[1] * h)))

    contour_px = [to_px(p) for p in geom["contour"]]
    tzone_px = [to_px(p) for p in geom["t_zone"]]
    lcheek_px = [to_px(p) for p in geom["left_cheek_zone"]]
    rcheek_px = [to_px(p) for p in geom["right_cheek_zone"]]

    draw.polygon(tzone_px, fill=(30, 108, 239, 44), outline=(37, 99, 235, 220))
    draw.polygon(lcheek_px, fill=(14, 165, 233, 38), outline=(56, 189, 248, 210))
    draw.polygon(rcheek_px, fill=(14, 165, 233, 38), outline=(56, 189, 248, 210))

    line_w = max(2, int(round(min(w, h) * 0.006)))
    draw.polygon(contour_px, outline=(255, 255, 255, 215))
    for i in range(len(contour_px)):
        p1 = contour_px[i]
        p2 = contour_px[(i + 1) % len(contour_px)]
        draw.line([p1, p2], fill=(30, 108, 239, 230), width=line_w)

    lm = geom["landmarks"]
    fh = to_px(lm["forehead"])
    le = to_px(lm["left_eye"])
    re = to_px(lm["right_eye"])
    nt = to_px(lm["nose_tip"])
    lc = to_px(lm["left_cheek"])
    rc = to_px(lm["right_cheek"])
    mc = to_px(lm["mouth_center"])
    ch = to_px(lm["chin"])

    mesh_edges = [
        (fh, le), (fh, re), (fh, nt),
        (le, re), (le, nt), (re, nt),
        (le, lc), (re, rc), (lc, nt), (rc, nt),
        (lc, mc), (rc, mc), (nt, mc),
        (lc, ch), (rc, ch), (mc, ch),
        (contour_px[11], fh), (contour_px[1], fh),
        (contour_px[10], le), (contour_px[2], re),
        (contour_px[9], lc), (contour_px[3], rc),
        (contour_px[8], mc), (contour_px[4], mc),
        (contour_px[7], ch), (contour_px[5], ch),
    ]
    thin_w = max(1, line_w - 1)
    for a, b in mesh_edges:
        draw.line([a, b], fill=(191, 219, 254, 185), width=thin_w)

    r_outer = max(3, int(round(min(w, h) * 0.011)))
    r_inner = max(2, r_outer - 1)
    for key_pt in [fh, le, re, nt, lc, rc, mc, ch]:
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
    norm_rgb: np.ndarray, skin_mask: np.ndarray, geom: Dict[str, Any]
) -> Image.Image:
    """
    Computes a spatial ~384 micro-patch skin biomarker map masked tightly inside
    the detected face contour.
    """
    r = norm_rgb[:, :, 0]
    g = norm_rgb[:, :, 1]
    b = norm_rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    gray_base = np.stack([lum * 0.92, lum * 0.95, lum * 1.02], axis=2)
    base_rgba = Image.fromarray(np.clip(gray_base, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    w, h = base_rgba.size

    face_poly = Image.new("L", (w, h), 0)
    poly_draw = ImageDraw.Draw(face_poly)
    contour_px = [(int(round(p[0] * w)), int(round(p[1] * h))) for p in geom["contour"]]
    poly_draw.polygon(contour_px, fill=255)
    face_arr = np.array(face_poly) > 128

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = geom["bbox"]
    fx0 = int(bbox["x"] * w)
    fy0 = int(bbox["y"] * h)
    fx1 = int((bbox["x"] + bbox["width"]) * w)
    fy1 = int((bbox["y"] + bbox["height"]) * h)

    cols, rows = 18, 22
    cell_w = max(4.0, (fx1 - fx0) / float(cols))
    cell_h = max(4.0, (fy1 - fy0) / float(rows))

    face_cr = cr[fy0:max(fy0 + 1, fy1), fx0:max(fx0 + 1, fx1)]
    baseline_cr = float(np.median(face_cr)) if face_cr.size > 0 else 140.0

    for row in range(rows):
        for col in range(cols):
            cx0 = int(round(fx0 + col * cell_w))
            cy0 = int(round(fy0 + row * cell_h))
            cx1 = int(round(fx0 + (col + 1) * cell_w))
            cy1 = int(round(fy0 + (row + 1) * cell_h))
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            patch_mask = face_arr[cy0:cy1, cx0:cx1]
            if np.mean(patch_mask) < 0.45:
                continue

            patch_cr = float(np.mean(cr[cy0:cy1, cx0:cx1]))
            patch_std = float(np.std(lum[cy0:cy1, cx0:cx1]))
            stress = _clamp(((patch_cr - baseline_cr) / 14.0) * 0.65 + (patch_std / 24.0) * 0.35, 0.0, 1.0)

            if stress < 0.35:
                fill_col = (30, 108, 239, 95)
            elif stress < 0.62:
                fill_col = (14, 165, 233, 105)
            elif stress < 0.80:
                fill_col = (245, 158, 11, 118)
            else:
                fill_col = (239, 68, 68, 128)

            draw.rectangle([cx0, cy0, cx1 - 1, cy1 - 1], fill=fill_col, outline=(255, 255, 255, 70))

    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def run_skin_cv_pipeline(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes the full Python Computer Vision skin pipeline in pure NumPy + Pillow
    (no SciPy shared-library overhead on serverless cold start).
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_w, orig_h = pil_img.size
    work_img = pil_img.copy()
    work_img.thumbnail((480, 480), Image.Resampling.LANCZOS)
    rgb = np.array(work_img, dtype=np.uint8)

    norm_rgb = _normalize_lighting_and_dull_razor(rgb)
    skin_mask = _segment_skin_ycbcr_hsv(norm_rgb)
    geom = _detect_face_geometry(norm_rgb, skin_mask)

    wood_img, wood_metrics = _generate_woodlamp_image(norm_rgb, skin_mask)
    mesh_img = _generate_geometry_mesh_image(norm_rgb, geom)
    micro_img = _generate_micro_analysis_image(norm_rgb, skin_mask, geom)

    skin_cov_pct = round(float(np.mean(skin_mask) * 100.0), 1)
    cv_metrics = {
        **wood_metrics,
        "skin_coverage_pct": skin_cov_pct,
        "image_width": orig_w,
        "image_height": orig_h,
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
        "woodlamp_image": _encode_jpeg_data_url(wood_img, max_width=380, quality=80),
        "geometry_mesh_image": _encode_jpeg_data_url(mesh_img, max_width=380, quality=82),
        "micro_analysis_image": _encode_jpeg_data_url(micro_img, max_width=380, quality=82),
        "cv_metrics": cv_metrics,
        "woods_lamp_findings": findings,
    }
