import os
import io
import re
import json
import base64
from typing import Optional, Dict, Any, List
from PIL import Image
from groq import AsyncGroq
from api.scin_taxonomy import (
    FITZPATRICK_TYPES,
    SCIN_TEXTURES,
    SCIN_SYMPTOMS,
    SCIN_BODY_PARTS,
    ENVIRONMENTAL_SCIN_CONDITIONS,
)

# User configured primary vision model: qwen/qwen3.8-27b
GROQ_DEFAULT_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_FALLBACK_VISION_MODEL = "llama-3.2-11b-vision-preview"

SCIN_SYSTEM_PROMPT = f"""
Anda adalah dermatologis AI ahli spesialis Dermatologi Lingkungan (Environmental Dermatology) yang menggunakan standar validasi dataset Google SCIN (Skin Condition Image Network).

Tugas Anda adalah menganalisis foto kondisi kulit pengguna yang berpotensi terkena dampak negatif dari 3 faktor lingkungan utama:
1. Paparan Polusi Udara (Particulate Matter PM2.5, hidrokarbon, debu mikro, penyumbatan pori, oksidasi sebum).
2. Cuaca & Iklim (Kelembapan ekstrem, cuaca panas memicu biang keringat, udara kering memicu kerusakan sawar kulit/TEWL).
3. Paparan Sinar Matahari / Radiasi Ultraviolet (UV) (Eritema matahari, sunburn, photodermatitis, hiperpigmentasi reaktif).

Validasi dan klasifikasikan temuan klinis Anda secara ketat mengacu pada taksonomi dataset SCIN berikut:

- FITZPATRICK SKIN TYPE: Pilih salah satu dari {list(FITZPATRICK_TYPES.keys())}.
- TEKSTUR SCIN: Pilih subset relevan dari {SCIN_TEXTURES}.
- GEJALA KLINIS SCIN: Pilih subset relevan dari {SCIN_SYMPTOMS}.
- AREA TUBUH (BODY PART): Pilih dari {SCIN_BODY_PARTS}.
- KONDISI KULIT SCIN LINGKUNGAN: Pilih kondisi yang paling mendekati dari:
  {list(ENVIRONMENTAL_SCIN_CONDITIONS.keys())}

Format output Anda WAJIB berupa JSON murni yang valid tanpa teks pembuka atau penutup selain JSON.

Struktur JSON yang wajib dipatuhi:
{{
  "condition": "<Nama kondisi dari daftar SCIN, misal: Photodermatitis / Eczema / Acne / Normal>",
  "condition_indonesian": "<Nama dan penjelasan ringkas dalam Bahasa Indonesia>",
  "confidence": <Angka 0 - 100>,
  "fitzpatrick_skin_type": "<FST1 - FST6>",
  "fitzpatrick_description": "<Deskripsi karakter kulit>",
  "scin_validation": {{
    "primary_condition": "<Nama kondisi SCIN>",
    "textures": ["<daftar tekstur SCIN yang teramati>"],
    "symptoms": ["<daftar gejala SCIN yang teramati atau potensial dirasakan>"],
    "body_part": "<area tubuh yang teramati, misal head_or_neck, arm, dll>",
    "is_scin_verified": true
  }},
  "environmental_impact": {{
    "pollution": {{
      "score": <0 - 100>,
      "level": "<Rendah | Sedang | Tinggi>",
      "analysis": "<Analisis dampak partikel polusi pada kulit teramati>"
    }},
    "weather": {{
      "score": <0 - 100>,
      "level": "<Rendah | Sedang | Tinggi>",
      "analysis": "<Analisis pengaruh kelembapan/suhu cuaca terhadap hidrasi kulit>"
    }},
    "sun_uv": {{
      "score": <0 - 100>,
      "level": "<Rendah | Sedang | Tinggi>",
      "analysis": "<Analisis dampak radiasi UV, kemerahan, atau pigmentasi>"
    }}
  }},
  "skin_metrics": {{
    "overall_health_score": <0 - 100, 100 = sangat sehat>,
    "hydration_score": <0 - 100>,
    "redness_score": <0 - 100, kemerahan/inflamasi>,
    "barrier_integrity_score": <0 - 100>,
    "pollution_clog_score": <0 - 100, tingkat komedo/polutan>,
    "uv_damage_score": <0 - 100, tingkat kerusakan radiasi UV>
  }},
  "severity": "<mild | moderate | severe | healthy>",
  "recommendations": [
    "<Rekomendasi langkah perawatan 1 (misal perlindungan sunscreen)>",
    "<Rekomendasi langkah perawatan 2 (misal pembersihan/double cleansing dari polutan)>",
    "<Rekomendasi langkah perawatan 3 (misal pelembap/perbaikan barier)>",
    "<Rekomendasi kapan harus konsultasi ke dokter spesialis kulit>"
  ],
  "summary": "<Ringkasan evaluasi kondisi kulit dalam 1-2 kalimat>",
  "medical_disclaimer": "Hasil analisis ini merupakan skrining berbasis AI dan dataset SCIN. Bukan pengganti diagnosis medis resmi dari dokter spesialis kulit."
}}
"""

def prepare_image_base64(image_bytes: bytes, max_size: int = 1024) -> str:
    """
    Validates and resizes image to reasonable dimensions for Groq Vision API.
    Returns base64 data URL string.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    # Resize if larger than max_size while maintaining aspect ratio
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """
    Strips code fences and extracts JSON dict reliably.
    """
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback regex search for json block { ... }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model output could not be parsed into JSON: {text[:200]}")

async def analyze_skin_with_groq(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    preferred_model: str = GROQ_DEFAULT_VISION_MODEL,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs multimodal skin condition inference using Groq AI with model qwen/qwen3.8-27b.
    Validates against SCIN dataset standards.
    """
    active_key = api_key or os.environ.get("GROQ_API_KEY")
    if not active_key:
        raise ValueError(
            "GROQ_API_KEY tidak ditemukan. Mohon sertakan API Key melalui environment variable "
            "atau header 'X-Groq-Api-Key'."
        )

    # Process image
    image_data_url = prepare_image_base64(image_bytes)

    # Build contextual user prompt with rich mobility & environmental data from aurabackend
    user_prompt = "Berikut adalah foto kondisi kulit untuk dianalisis terhadap dampak polusi, cuaca, dan radiasi matahari sesuai taksonomi SCIN."
    if context:
        user_prompt += "\n\nKonteks Lingkungan, Mobilitas, & Profil Pasien (dari AURA Backend):\n"
        
        # 1. Mobility tracking data
        if "total_outdoor_minutes" in context:
            user_prompt += f"- Durasi Paparan Luar Ruangan (Outdoor): {context['total_outdoor_minutes']} menit\n"
        if "total_indoor_minutes" in context:
            user_prompt += f"- Durasi Ruang Tertutup (Indoor): {context['total_indoor_minutes']} menit\n"
        if "average_outdoor_pm25" in context:
            user_prompt += f"- Rata-rata PM2.5 Paparan Luar: {context['average_outdoor_pm25']} µg/m³\n"
        elif "pm2_5" in context:
            user_prompt += f"- Konsentrasi PM2.5: {context['pm2_5']} µg/m³\n"
        if "average_outdoor_aqi" in context:
            user_prompt += f"- Rata-rata AQI Luar: {context['average_outdoor_aqi']}\n"
        elif "aqi" in context:
            user_prompt += f"- Indeks Kualitas Udara (AQI): {context['aqi']}\n"
        if "max_aqi" in context:
            user_prompt += f"- AQI Puncak Hari Ini: {context['max_aqi']}\n"
        if "worst_area_name" in context:
            user_prompt += f"- Titik Paparan Terburuk: {context['worst_area_name']} (AQI {context.get('worst_area_aqi', '-')}) pada {context.get('worst_area_timestamp', '-')}\n"
        if "city" in context:
            user_prompt += f"- Lokasi Kawasan: {context['city']}\n"

        # 2. Weather & multi-pollutant environmental factors
        if "temperature" in context:
            user_prompt += f"- Suhu Udara: {context['temperature']} °C\n"
        if "relative_humidity" in context or "humidity" in context:
            user_prompt += f"- Kelembapan Relatif: {context.get('relative_humidity', context.get('humidity'))}%\n"
        if "uv_index" in context:
            user_prompt += f"- Indeks Radiasi UV: {context['uv_index']}\n"
        if "weather" in context:
            user_prompt += f"- Kondisi Cuaca: {context['weather']}\n"
        if "wind_speed" in context:
            user_prompt += f"- Kecepatan Angin: {context['wind_speed']} km/jam\n"
        if "pm10" in context:
            user_prompt += f"- PM10 (Debu Kasar): {context['pm10']} µg/m³\n"
        if "nitrogen_dioxide" in context:
            user_prompt += f"- NO2 (Emisi Kendaraan): {context['nitrogen_dioxide']} µg/m³\n"
        if "carbon_monoxide" in context:
            user_prompt += f"- CO (Karbon Monoksida): {context['carbon_monoxide']} µg/m³\n"
        if "congestion_index" in context:
            user_prompt += f"- Indeks Kemacetan Lalu Lintas: {context['congestion_index']}/100\n"

        # 3. User physical vulnerability profile
        if "usia" in context or "age" in context:
            user_prompt += f"- Usia Pengguna: {context.get('usia', context.get('age'))} tahun\n"
        if "tinggi_badan" in context and "berat_badan" in context:
            tb = float(context["tinggi_badan"]) / 100.0 if float(context["tinggi_badan"]) > 3 else float(context["tinggi_badan"])
            bb = float(context["berat_badan"])
            bmi = round(bb / (tb * tb), 1) if tb > 0 else 0
            user_prompt += f"- Profil Fisik: TB {context['tinggi_badan']} cm, BB {context['berat_badan']} kg (BMI ~{bmi})\n"

    active_base_url = base_url or os.environ.get("GROQ_BASE_URL")
    client_kwargs: Dict[str, Any] = {"api_key": active_key}
    if active_base_url:
        client_kwargs["base_url"] = active_base_url

    client = AsyncGroq(**client_kwargs)

    messages = [
        {"role": "system", "content": SCIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    response_text = ""
    models_to_try = [preferred_model]
    if preferred_model != GROQ_FALLBACK_VISION_MODEL:
        models_to_try.append(GROQ_FALLBACK_VISION_MODEL)

    last_err = None
    for model_name in models_to_try:
        try:
            # Try first with response_format json_object
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # If json_object mode not supported on specific model, retry standard completion
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                )
            response_text = completion.choices[0].message.content
            break
        except Exception as err:
            last_err = err
            continue

    if not response_text and last_err:
        raise last_err

    # Parse and validate JSON
    parsed = parse_json_safely(response_text)
    
    # Attach model information to response
    parsed["model_used"] = model_name

    # Enrich with SCIN taxonomy reference if present
    cond_name = parsed.get("condition")
    if cond_name in ENVIRONMENTAL_SCIN_CONDITIONS:
        ref = ENVIRONMENTAL_SCIN_CONDITIONS[cond_name]
        parsed["scin_taxonomy_reference"] = {
            "canonical_label": ref["scin_label"],
            "primary_category": ref["category"],
            "official_description": ref["description"]
        }
        
    fst = parsed.get("fitzpatrick_skin_type")
    if fst in FITZPATRICK_TYPES:
        parsed["fitzpatrick_details"] = FITZPATRICK_TYPES[fst]

    return parsed
