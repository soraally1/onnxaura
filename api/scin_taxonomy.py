"""
SCIN Dataset Taxonomy and Environmental Dermatology Mapping
Based on Google's SCIN (Skin Condition Image Network) dataset:
https://github.com/google-research-datasets/scin
"""

from typing import List, Dict, Any

# Fitzpatrick Skin Types (from SCIN dataset)
FITZPATRICK_TYPES = {
    "FST1": {
        "name": "Tipe I",
        "description": "Sangat putih / pucat, selalu terbakar matahari, tidak pernah kecokelatan (tanning)",
        "sun_sensitivity": "Sangat Tinggi"
    },
    "FST2": {
        "name": "Tipe II",
        "description": "Putih, biasanya mudah terbakar matahari, sedikit mengalami tanning",
        "sun_sensitivity": "Tinggi"
    },
    "FST3": {
        "name": "Tipe III",
        "description": "Putih ke sawo matang muda, kadang terbakar ringan, mengalami tanning bertahap",
        "sun_sensitivity": "Sedang"
    },
    "FST4": {
        "name": "Tipe IV",
        "description": "Sawo matang (khas Indonesia / Mediterania), jarang terbakar, mudah tanning",
        "sun_sensitivity": "Rendah ke Sedang"
    },
    "FST5": {
        "name": "Tipe V",
        "description": "Cokelat gelap, sangat jarang terbakar, sangat mudah mengalami pigmentasi pekat",
        "sun_sensitivity": "Rendah"
    },
    "FST6": {
        "name": "Tipe VI",
        "description": "Cokelat sangat gelap / hitam, tidak pernah terbakar, pigmentasi sangat pekat",
        "sun_sensitivity": "Sangat Rendah"
    }
}

# SCIN Observable Textures (from SCIN dataset)
SCIN_TEXTURES = [
    "raised_or_bumpy",
    "flat",
    "rough_or_flaky",
    "fluid_filled"
]

# SCIN Observable Symptoms (from SCIN dataset)
SCIN_SYMPTOMS = [
    "itching",
    "burning",
    "pain",
    "darkening",
    "bleeding",
    "increasing_size",
    "bothersome_appearance"
]

# SCIN Body Parts (from SCIN dataset)
SCIN_BODY_PARTS = [
    "head_or_neck",
    "arm",
    "palm",
    "back_of_hand",
    "torso_front",
    "torso_back",
    "leg",
    "foot_top_or_side",
    "foot_sole",
    "other"
]

# Environmental Dermatological Conditions mapped to SCIN taxonomy
ENVIRONMENTAL_SCIN_CONDITIONS = {
    # ── Sun / UV Exposure ──
    "Photodermatitis": {
        "indonesian": "Fotodermatitis (Sensitivitas / Alergi Sinar Matahari)",
        "category": "sun_uv",
        "scin_label": "Photodermatitis",
        "description": "Reaksi inflamasi abnormal pada kulit akibat paparan sinar radiasi ultraviolet (UV)."
    },
    "Post-Inflammatory hyperpigmentation": {
        "indonesian": "Hiperpigmentasi Pasca-Inflamasi (Noda Gelap UV / Polusi)",
        "category": "sun_uv",
        "scin_label": "Post-Inflammatory hyperpigmentation",
        "description": "Bercak gelap akibat produksi melanin berlebih pasca inflamasi yang dipicu radiasi UV matahari dan polutan."
    },
    "Actinic damage / Sunburn": {
        "indonesian": "Eritema Matahari / Kerusakan Aktinik (Sunburn)",
        "category": "sun_uv",
        "scin_label": "Photodermatitis",
        "description": "Kemerahan akut dan kerusakan sel epidermis akibat radiasi UV tanpa proteksi."
    },

    # ── Weather / Climate Exposure ──
    "Eczema": {
        "indonesian": "Eksim / Dermatitis Atopik",
        "category": "weather",
        "scin_label": "Eczema",
        "description": "Peradangan barier kulit kronis yang kambuh akibat perubahan cuaca ekstrem, udara kering, atau kelembapan tinggi."
    },
    "Irritant Contact Dermatitis": {
        "indonesian": "Dermatitis Kontak Iritan",
        "category": "weather",
        "scin_label": "Irritant Contact Dermatitis",
        "description": "Iritasi sawar kulit akibat angin dingin, keringat berlebih, atau bahan kimia iritan lingkungan."
    },
    "Miliaria": {
        "indonesian": "Miliaria (Biang Keringat / Ruam Panas)",
        "category": "weather",
        "scin_label": "Miliaria",
        "description": "Penyumbatan saluran kelenjar keringat akibat cuaca tropis yang panas dan lembap."
    },
    "Rosacea": {
        "indonesian": "Rosacea",
        "category": "weather",
        "scin_label": "Rosacea",
        "description": "Kemerahan kronis pada wajah yang mudah terpicu oleh paparan cuaca panas, angin kencang, dan sinar matahari."
    },

    # ── Air Pollution Exposure ──
    "Acne": {
        "indonesian": "Jerawat / Komedo Akibat Polusi",
        "category": "pollution",
        "scin_label": "Acne",
        "description": "Penyumbatan folikel rambut dan pori-pori oleh partikel debu halus (PM2.5) serta oksidasi sebum."
    },
    "Folliculitis": {
        "indonesian": "Folikulitis (Infeksi / Inflamasi Pori-Pori)",
        "category": "pollution",
        "scin_label": "Folliculitis",
        "description": "Peradangan pada folikel rambut yang diperparah oleh polutan mikro dan penumpukan debu/keringat."
    },
    "Allergic Contact Dermatitis": {
        "indonesian": "Dermatitis Kontak Alergi",
        "category": "pollution",
        "scin_label": "Allergic Contact Dermatitis",
        "description": "Reaksi alergi hipersensitivitas terhadap alergen udara atau partikel polusi lingkungan."
    },
    "Urticaria": {
        "indonesian": "Urtikaria (Biduran / Kaligata)",
        "category": "pollution",
        "scin_label": "Urticaria",
        "description": "Bentol gatal akut yang dapat terpicu oleh polusi kimia, perubahan temperatur, atau radiasi matahari."
    },
    "Acute dermatitis, NOS": {
        "indonesian": "Dermatitis Akut Non-Spesifik",
        "category": "pollution",
        "scin_label": "Acute dermatitis, NOS",
        "description": "Reaksi kemerahan dan iritasi akut akibat paparan agresor eksternal lingkungan."
    },
    "Normal": {
        "indonesian": "Kulit Sehat (Barier Optimal)",
        "category": "healthy",
        "scin_label": "Normal",
        "description": "Kondisi kulit sehat dengan integritas barier stratum korneum yang baik tanpa inflamasi signifikan."
    }
}
