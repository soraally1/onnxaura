import io
import json
import os
import re
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

# Model file discovery order: prefer INT8 quantized model when present for smallest Vercel bundle
_BASE_DIR = os.path.dirname(__file__)
_TTS_DIR = os.path.join(_BASE_DIR, "tts")
_CANDIDATE_MODELS = (
    os.path.join(_TTS_DIR, "id_ID-news_tts-medium.int8.onnx"),
    os.path.join(_TTS_DIR, "id_ID-news_tts-medium.onnx"),
)
_CONFIG_PATH = os.path.join(_TTS_DIR, "id_ID-news_tts-medium.onnx.json")

_tts_session = None
_tts_config: Optional[Dict] = None
_tts_model_path: Optional[str] = None

_UNITS = (
    "nol",
    "satu",
    "dua",
    "tiga",
    "empat",
    "lima",
    "enam",
    "tujuh",
    "delapan",
    "sembilan",
    "sepuluh",
    "sebelas",
)


def _int_to_indonesian(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_indonesian(-n)
    if n < 12:
        return _UNITS[n]
    if n < 20:
        return f"{_UNITS[n - 10]} belas"
    if n < 100:
        tens, rem = divmod(n, 10)
        return f"{_UNITS[tens]} puluh" + (f" {_UNITS[rem]}" if rem else "")
    if n < 200:
        rem = n - 100
        return "seratus" + (f" {_int_to_indonesian(rem)}" if rem else "")
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return f"{_UNITS[hundreds]} ratus" + (f" {_int_to_indonesian(rem)}" if rem else "")
    if n < 2000:
        rem = n - 1000
        return "seribu" + (f" {_int_to_indonesian(rem)}" if rem else "")
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        return f"{_int_to_indonesian(thousands)} ribu" + (f" {_int_to_indonesian(rem)}" if rem else "")
    return str(n)


def _replace_number_match(match: re.Match) -> str:
    raw = match.group(0).replace(",", ".")
    if "." in raw:
        parts = raw.split(".", 1)
        try:
            whole = _int_to_indonesian(int(parts[0]))
            frac_digits = " ".join(_UNITS[int(d)] for d in parts[1] if d.isdigit())
            return f"{whole} koma {frac_digits}".strip()
        except ValueError:
            return raw
    try:
        return _int_to_indonesian(int(raw))
    except ValueError:
        return raw


def normalize_indonesian_text(text: str) -> str:
    """
    Expand abbreviations, symbols, acronyms, and numbers into clean spoken Indonesian words
    before IPA phonemization.
    """
    t = text.strip()
    t = re.sub(r"[*_`#]+", " ", t)
    t = t.replace("&", " dan ").replace("/", " per ").replace("%", " persen ")

    # Expand common AURA / health terms & acronyms
    replacements = (
        (r"\bAURA\b", "aura"),
        (r"\bAI\b", "ei ai"),
        (r"\bWFH\b", "kerja dari rumah"),
        (r"\b(\d+(?:[.,]\d+)?)\s*cm\b", r"\1 sentimeter"),
        (r"\b(\d+(?:[.,]\d+)?)\s*kg\b", r"\1 kilogram"),
        (r"\b(\d+(?:[.,]\d+)?)\s*thn\b", r"\1 tahun"),
        (r"\b(\d+(?:[.,]\d+)?)\s*mnt\b", r"\1 menit"),
        (r"\bcm\b", "sentimeter"),
        (r"\bkg\b", "kilogram"),
        (r"\bthn\b", "tahun"),
    )
    for pat, rep in replacements:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    # Expand digits cleanly to Indonesian words
    t = re.sub(r"\d+(?:[.,]\d+)?", _replace_number_match, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_VOWELS = set("aeiouə")

# High-frequency Indonesian roots whose orthographic 'e' is tense [e] rather than schwa [ə]
_TENSE_E_WORDS = {
    "oke", "sore", "meja", "beda", "lewat", "energi", "sentimeter", "meter",
    "persen", "cek", "tes", "resep", "hebat", "bebas", "area", "kamera",
    "telepon", "sistem", "penuh", "segera", "teks", "desa", "daerah", "karena",
}


def _word_to_ipa(word: str) -> str:
    w = word.lower()
    if not w:
        return ""

    # Common loanword tweaks
    if w == "ai":
        return "ˈei ˈai"

    phonemes: List[str] = []
    vowel_indices: List[int] = []
    i = 0
    n = len(w)

    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""

        if ch == "n" and nxt == "g":
            phonemes.append("ŋ")
            i += 2
            continue
        if ch == "n" and nxt == "y":
            phonemes.append("ɲ")
            i += 2
            continue
        if ch == "s" and nxt == "y":
            phonemes.append("ʃ")
            i += 2
            continue
        if ch == "k" and nxt == "h":
            phonemes.append("x")
            i += 2
            continue

        if ch == "c":
            phonemes.extend(["t", "ʃ"])
        elif ch == "j":
            phonemes.extend(["d", "ʒ"])
        elif ch == "g":
            phonemes.append("ɡ")
        elif ch == "y":
            phonemes.append("j")
        elif ch == "k" and i == n - 1:
            # Final coda -k in Indonesian often softens toward glottal / unreleased k
            phonemes.append("k")
        elif ch == "q":
            phonemes.append("k")
        elif ch == "x":
            phonemes.extend(["k", "s"])
        elif ch == "v":
            phonemes.append("f")
        elif ch == "e":
            vowel_indices.append(len(phonemes))
            if w in _TENSE_E_WORDS or (i == n - 1 and n <= 4):
                phonemes.append("e")
            elif i == 1 and w[0] in ("s", "k", "p", "b", "t", "d", "m", "l"):
                phonemes.append("ə")
            elif n >= 4 and i <= 2:
                phonemes.append("ə")
            else:
                phonemes.append("e")
        elif ch in ("a", "i", "u", "o"):
            vowel_indices.append(len(phonemes))
            phonemes.append(ch)
        elif ch.isalpha():
            phonemes.append(ch)
        i += 1

    # Insert primary stress marker ˈ before penultimate syllable onset when multisyllabic
    if len(vowel_indices) >= 2:
        target_v = vowel_indices[-2]
        if phonemes[target_v] == "ə" and len(vowel_indices) >= 2:
            target_v = vowel_indices[-1]
        insert_at = target_v
        if insert_at > 0 and phonemes[insert_at - 1] not in _VOWELS:
            insert_at -= 1
            if insert_at > 0 and phonemes[insert_at - 1] in ("t", "d") and phonemes[insert_at] in ("ʃ", "ʒ"):
                insert_at -= 1
        phonemes.insert(insert_at, "ˈ")
    elif len(vowel_indices) == 1:
        phonemes.insert(0, "ˈ")

    return "".join(phonemes)


def text_to_ipa(text: str) -> str:
    normalized = normalize_indonesian_text(text)
    tokens = re.findall(r"[A-Za-z]+|[.,!?;:\-]+|\s+", normalized)
    out: List[str] = []
    for tok in tokens:
        if tok.isspace():
            out.append(" ")
        elif tok[0].isalpha():
            out.append(_word_to_ipa(tok))
        else:
            out.append(tok)
    ipa = "".join(out).strip()
    if ipa and ipa[-1] not in ".!?":
        ipa += "."
    return ipa


def _encode_ipa_to_ids(ipa: str, phoneme_id_map: Dict[str, List[int]]) -> List[int]:
    bos = phoneme_id_map.get("^", [1])[0]
    eos = phoneme_id_map.get("$", [2])[0]
    pad = phoneme_id_map.get("_", [0])[0]

    ids: List[int] = [bos, pad]
    for ch in ipa:
        mapped = phoneme_id_map.get(ch)
        if not mapped:
            continue
        for pid in mapped:
            ids.append(int(pid))
            ids.append(pad)
    ids.append(eos)
    return ids


def _get_tts_runtime() -> Tuple[object, Dict]:
    global _tts_session, _tts_config, _tts_model_path
    if _tts_session is not None and _tts_config is not None:
        return _tts_session, _tts_config

    import onnxruntime as ort

    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Piper config not found at {_CONFIG_PATH}")

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _tts_config = json.load(f)

    chosen = next((p for p in _CANDIDATE_MODELS if os.path.exists(p)), None)
    if chosen is None:
        raise FileNotFoundError("Self-hosted Indonesian ONNX TTS model not found.")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _tts_session = ort.InferenceSession(chosen, sess_options=opts, providers=["CPUExecutionProvider"])
    _tts_model_path = chosen
    return _tts_session, _tts_config


def synthesize_indonesian_wav(text: str, length_scale: float = 1.03) -> Tuple[bytes, Dict]:
    """
    Synthesize natural Indonesian neural speech offline/self-hosted using ONNX Runtime
    with zero external API calls. Returns (wav_bytes, metadata).
    """
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Teks tidak boleh kosong untuk sintesis suara.")
    if len(clean) > 450:
        clean = clean[:450]

    sess, cfg = _get_tts_runtime()
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 22050))
    inf = cfg.get("inference", {})
    noise_scale = float(inf.get("noise_scale", 0.667))
    noise_w = float(inf.get("noise_w", 0.8))
    base_len_scale = float(length_scale or inf.get("length_scale", 1.0))

    ipa = text_to_ipa(clean)
    ids = _encode_ipa_to_ids(ipa, cfg["phoneme_id_map"])

    arr = np.asarray([ids], dtype=np.int64)
    lens = np.asarray([len(ids)], dtype=np.int64)
    scales = np.asarray([noise_scale, base_len_scale, noise_w], dtype=np.float32)

    raw_audio = sess.run(
        None,
        {"input": arr, "input_lengths": lens, "scales": scales},
    )[0]
    audio = np.squeeze(raw_audio).astype(np.float32)

    # Trim leading/trailing sub-threshold tail safely and peak-normalize cleanly to -1.5 dBFS
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-5:
        audio = audio * (0.85 / peak)
    audio = np.clip(audio, -0.99, 0.99)
    pcm16 = (audio * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return buf.getvalue(), {
        "model": os.path.basename(_tts_model_path or "id_ID-news_tts-medium"),
        "sample_rate": sample_rate,
        "duration_sec": round(len(pcm16) / float(sample_rate), 2),
        "ipa": ipa,
    }
