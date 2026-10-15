import os
import json
import base64
from typing import Optional, Dict, Any

# Configure environment directories for serverless environments (e.g. Vercel)
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['XDG_CACHE_HOME'] = '/tmp'

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import api.dsp as dsp
from scipy.special import softmax

from api.skin_analysis import analyze_skin_with_groq
from api.scin_taxonomy import (
    FITZPATRICK_TYPES,
    SCIN_TEXTURES,
    SCIN_SYMPTOMS,
    SCIN_BODY_PARTS,
    ENVIRONMENTAL_SCIN_CONDITIONS,
)

app = FastAPI(
    title="AURA Health AI - Cough & Environmental Skin Analysis API",
    description="API for respiratory cough analysis (ONNX ICBHI) and environmental skin scan (Groq AI + Google SCIN validation).",
    version="2.0.0"
)

# Enable CORS for Mobile App and Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- COUGH MODEL CLASS DEFINITIONS (ICBHI) ---
# Model used: best_icbhi_model.onnx
ICBHI_CLASSES = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Crackle + Wheeze"}

COUGH_CLINICAL_MAPPING = {
    "Normal": {
        "status": "healthy",
        "description": "Suara batuk dan pola pernapasan normal tanpa suara adventisius (crackle atau wheeze).",
        "severity": "low",
        "score": 92
    },
    "Crackle": {
        "status": "warning",
        "description": "Terdeteksi suara crackle (ronki basah), mengindikasikan adanya cairan, sekret, atau inflamasi pada saluran napas kecil.",
        "severity": "moderate",
        "score": 65
    },
    "Wheeze": {
        "status": "warning",
        "description": "Terdeteksi suara wheezing (mengi), mengindikasikan penyempitan jalan napas atau bronkospasme akibat iritasi polusi udara/asma.",
        "severity": "moderate",
        "score": 58
    },
    "Crackle + Wheeze": {
        "status": "warning",
        "description": "Terdeteksi kombinasi crackle dan wheeze, menunjukkan obstruksi dan penumpukan sekret signifikan pada saluran pernapasan.",
        "severity": "high",
        "score": 42
    }
}

# --- LOAD COUGH MODEL ONLY ---
print("Loading Cough ONNX Model (best_icbhi_model.onnx)...")
icbhi_path = os.path.join(os.path.dirname(__file__), "best_icbhi_model.onnx")

if not os.path.exists(icbhi_path):
    raise FileNotFoundError(f"Cough ONNX model not found at {icbhi_path}")

cough_session = ort.InferenceSession(icbhi_path)
print("Cough ONNX Model loaded successfully.")


# =====================================================================
# ROOT & HEALTH CHECK
# =====================================================================
@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "AURA Health AI",
        "active_models": {
            "cough_model": "best_icbhi_model.onnx (Loaded)",
            "skin_model": "Groq AI (qwen/qwen3.8-27b) + Google SCIN Validation"
        },
        "endpoints": {
            "cough_analysis": "POST /api/predict/cough",
            "skin_scan_form": "POST /api/skin/scan",
            "skin_scan_json": "POST /api/skin/scan-json",
            "scin_metadata": "GET /api/scin/metadata"
        }
    }


# =====================================================================
# COUGH / RESPIRATORY ANALYSIS ENDPOINT
# =====================================================================
async def process_cough_inference(
    file: UploadFile,
    total_outdoor_minutes: Optional[float] = None,
    average_outdoor_pm25: Optional[float] = None,
    worst_area_name: Optional[str] = None,
    worst_area_aqi: Optional[int] = None,
    usia: Optional[int] = None
):
    if not (file.filename.endswith('.wav') or file.filename.endswith('.WAV')):
        raise HTTPException(status_code=400, detail="Only .wav audio files are supported for cough analysis.")
        
    try:
        audio_bytes = await file.read()
        mel_spec = dsp.process_audio(audio_bytes) 
        
        max_length = 128
        if mel_spec.shape[1] > max_length:
            mel_spec = mel_spec[:, :max_length]
        else:
            pad_width = max_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            
        # Reshape to (1, 1, 128, 128) and cast to float32
        mel_array = np.array(mel_spec, dtype=np.float32).reshape(1, 1, 128, 128)
        
        # ONNX Inference
        input_name = cough_session.get_inputs()[0].name
        outputs = cough_session.run(None, {input_name: mel_array})
        
        # Calculate probabilities using Softmax
        logits = outputs[0][0] # shape (4,)
        probabilities = softmax(logits)
            
        pred_class_idx = int(np.argmax(probabilities))
        pred_label = ICBHI_CLASSES[pred_class_idx]
        confidence = float(probabilities[pred_class_idx])
        
        clinical_info = COUGH_CLINICAL_MAPPING.get(pred_label, {})
        base_score = clinical_info.get("score", 70)
        status = clinical_info.get("status", "moderate")
        severity = clinical_info.get("severity", "moderate")
        interpretation = clinical_info.get("description", "")

        # Correlate with Backend Mobility Tracking & Pollution exposure
        correlation = None
        if average_outdoor_pm25 is not None or total_outdoor_minutes is not None:
            pm25_val = float(average_outdoor_pm25 or 0)
            outdoor_mins = float(total_outdoor_minutes or 0)
            
            # High pollution impact threshold (WHO standard PM2.5 > 25 µg/m³ 24h, unhealthy > 55 µg/m³)
            is_high_exposure = pm25_val >= 55 or outdoor_mins >= 90
            
            risk_notes = []
            if pred_label in ["Wheeze", "Crackle", "Crackle + Wheeze"]:
                if is_high_exposure:
                    status = "warning"
                    base_score = max(30, base_score - 15)
                    risk_notes.append(
                        f"Gejala {pred_label} berkorelasi kuat dengan paparan polusi outdoor ({outdoor_mins} menit, rata-rata PM2.5: {pm25_val} µg/m³)."
                    )
                    if worst_area_name:
                        risk_notes.append(f"Titik paparan terburuk terjadi di kawasan {worst_area_name} (AQI {worst_area_aqi or '-'}).")
                else:
                    risk_notes.append(f"Suara {pred_label} terdeteksi meski paparan polusi relatif rendah. Perlu perhatian riwayat alergi atau infeksi pernapasan.")
            else:
                if is_high_exposure:
                    risk_notes.append(f"Pola batuk saat ini normal, namun Anda mengalami paparan polusi tinggi ({outdoor_mins} menit di luar). Tetap jaga hidrasi dan kenakan masker pelindung.")

            correlation = {
                "total_outdoor_minutes": outdoor_mins,
                "average_outdoor_pm25": pm25_val,
                "worst_area_name": worst_area_name,
                "worst_area_aqi": worst_area_aqi,
                "high_exposure_flag": is_high_exposure,
                "clinical_correlation_notes": " ".join(risk_notes)
            }
        
        response_data = {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "status": status,
            "health_score": base_score,
            "severity": severity,
            "clinical_interpretation": interpretation,
            "all_probabilities": {
                label: round(float(probabilities[idx]) * 100, 2) 
                for idx, label in ICBHI_CLASSES.items()
            }
        }
        if correlation:
            response_data["environmental_correlation"] = correlation

        return response_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing cough audio: {str(e)}")

@app.post("/api/predict/cough")
async def predict_cough(
    file: UploadFile = File(...),
    total_outdoor_minutes: Optional[float] = Form(None),
    average_outdoor_pm25: Optional[float] = Form(None),
    worst_area_name: Optional[str] = Form(None),
    worst_area_aqi: Optional[int] = Form(None),
    usia: Optional[int] = Form(None),
):
    """
    Primary endpoint for cough audio analysis.
    Optionally correlates findings with AURA Backend Mobility Tracking and PM2.5 exposure.
    """
    return await process_cough_inference(
        file=file,
        total_outdoor_minutes=total_outdoor_minutes,
        average_outdoor_pm25=average_outdoor_pm25,
        worst_area_name=worst_area_name,
        worst_area_aqi=worst_area_aqi,
        usia=usia
    )

@app.post("/api/predict/icbhi")
async def predict_icbhi(file: UploadFile = File(...)):
    """Backward compatibility alias for cough analysis."""
    return await process_cough_inference(file)

@app.post("/api/predict")
async def predict_root(file: UploadFile = File(...)):
    """Default predict alias route."""
    return await process_cough_inference(file)


# =====================================================================
# GROQ AI SKIN ANALYSIS WITH SCIN VALIDATION
# =====================================================================
@app.post("/api/skin/scan")
async def scan_skin(
    file: UploadFile = File(...),
    city: Optional[str] = Form(None),
    aqi: Optional[float] = Form(None),
    pm2_5: Optional[float] = Form(None),
    weather: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    uv_index: Optional[float] = Form(None),
    model: Optional[str] = Form(None),
    x_groq_api_key: Optional[str] = Header(None, alias="X-Groq-Api-Key")
):
    """
    Multimodal Skin Scan using Groq AI Vision model (qwen/qwen3.8-27b) validated with Google's SCIN dataset.
    Analyzes skin impact from air pollution, weather, and sun/UV exposure.
    """
    valid_content_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type and file.content_type.lower() not in valid_content_types and not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Mohon gunakan format gambar JPEG, PNG, atau WebP."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="File gambar kosong.")

        context = {}
        if city:
            context["city"] = city
        if aqi is not None:
            context["aqi"] = aqi
        if pm2_5 is not None:
            context["pm2_5"] = pm2_5
        if weather:
            context["weather"] = weather
        if temperature is not None:
            context["temperature"] = temperature
        if uv_index is not None:
            context["uv_index"] = uv_index

        kwargs = {
            "image_bytes": image_bytes,
            "api_key": x_groq_api_key,
            "context": context if context else None
        }
        if model:
            kwargs["preferred_model"] = model

        result = await analyze_skin_with_groq(**kwargs)
        return {
            "success": True,
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan analisis kulit dengan Groq AI: {str(e)}")


@app.post("/api/skin/scan-json")
async def scan_skin_json(
    payload: Dict[str, Any] = Body(...),
    x_groq_api_key: Optional[str] = Header(None, alias="X-Groq-Api-Key")
):
    """
    JSON endpoint for base64 encoded image skin analysis.
    Payload: { "image_base64": "...", "context": { "city": "...", "aqi": 120, ... }, "model": "qwen/qwen3.8-27b" }
    """
    image_base64 = payload.get("image_base64") or payload.get("image")
    if not image_base64:
        raise HTTPException(status_code=400, detail="Field 'image_base64' wajib diisi.")

    try:
        # Strip data URL prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
            
        image_bytes = base64.b64decode(image_base64)
        context = payload.get("context", {})

        kwargs = {
            "image_bytes": image_bytes,
            "api_key": x_groq_api_key or payload.get("groq_api_key"),
            "context": context
        }
        if payload.get("model"):
            kwargs["preferred_model"] = payload.get("model")

        result = await analyze_skin_with_groq(**kwargs)
        return {
            "success": True,
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menganalisis gambar base64: {str(e)}")


# =====================================================================
# SCIN METADATA & TAXONOMY REFERENCE
# =====================================================================
@app.get("/api/scin/metadata")
def get_scin_metadata():
    """
    Returns reference taxonomy from the Google SCIN dataset
    for environmental skin screening.
    """
    return {
        "dataset_name": "Google SCIN (Skin Condition Image Network)",
        "fitzpatrick_types": FITZPATRICK_TYPES,
        "scin_textures": SCIN_TEXTURES,
        "scin_symptoms": SCIN_SYMPTOMS,
        "scin_body_parts": SCIN_BODY_PARTS,
        "environmental_conditions": ENVIRONMENTAL_SCIN_CONDITIONS
    }
