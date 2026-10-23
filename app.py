import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_KEY = "cnn_scratch"  
ALLOWED_TYPES = ["jpeg", "jpg", "png", "bmp"]
MAX_MB = 10

st.set_page_config(page_title="ChestScan AI", page_icon="🩻", layout="centered")

def available_models():
    return sorted(
        p.stem for p in MODELS_DIR.glob("*.json") if (MODELS_DIR / f"{p.stem}.keras").exists()
    )


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(key: str):
    from tensorflow import keras

    cfg = json.loads((MODELS_DIR / f"{key}.json").read_text())
    model = keras.models.load_model(MODELS_DIR / f"{key}.keras", compile=False)
    return model, cfg


def read_image(uploaded) -> Image.Image:
    """Décode et valide le fichier. Lève ValueError avec un message lisible."""
    if uploaded.size > MAX_MB * 1024 * 1024:
        raise ValueError(f"Fichier trop volumineux (> {MAX_MB} Mo).")
    try:
        img = Image.open(io.BytesIO(uploaded.getvalue()))
        img.load()  
        return img.convert("RGB") 
    except Exception as e:
        raise ValueError(f"Image illisible ou corrompue ({type(e).__name__}).") from e


def preprocess(img: Image.Image, cfg: dict):
    """Même chaîne que predict_xray : resize bilinéaire (tf.image.resize) puis
    preprocess_input si le modèle l'exige (MobileNetV2). Le CNN scratch
    contient déjà Rescaling(1/255)."""
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    x = tf.image.resize(np.asarray(img, dtype="float32"), tuple(cfg["img_size"]))
    x = tf.expand_dims(x, 0)
    if cfg["preprocess"] == "mobilenet_v2":
        x = preprocess_input(x)
    return x


def predict(img: Image.Image, model, cfg: dict):
    score = float(model.predict(preprocess(img, cfg), verbose=0)[0, 0])  
    label = cfg["class_names"][int(score >= cfg["threshold"])]
    return label, score


st.title("🩻 ChestScan AI")
st.warning(
    "**Prototype pédagogique – non destiné au diagnostic médical.** "
    "Les résultats ne remplacent en aucun cas l'avis d'un professionnel de santé."
)

keys = available_models()
if not keys:
    st.error(f"Aucun modèle trouvé dans `{MODELS_DIR}` (fichiers .keras + .json attendus).")
    st.stop()

with st.sidebar:
    st.header("Modèle")
    model_key = st.selectbox(
        "Modèle utilisé", keys, index=keys.index(DEFAULT_KEY) if DEFAULT_KEY in keys else 0
    )

uploaded = st.file_uploader(
    "Importer une radiographie thoracique", type=ALLOWED_TYPES, accept_multiple_files=False
)

if uploaded is None:
    st.info("Formats acceptés : JPEG, JPG, PNG, BMP.")
    st.stop()

try:
    image = read_image(uploaded)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.image(image, caption=f"{uploaded.name} – {image.width}×{image.height}px", use_container_width=True)

if st.button("Lancer la prédiction", type="primary"):
    try:
        model, cfg = load_model(model_key)
    except Exception as e:
        st.error(f"Impossible de charger le modèle « {model_key} » : {type(e).__name__}: {e}")
        st.stop()
    try:
        with st.spinner("Analyse en cours…"):
            label, score = predict(image, model, cfg)
    except Exception as e:
        st.error(f"Échec de la prédiction : {type(e).__name__}: {e}")
        st.stop()

    st.session_state["result"] = {
        "sig": (uploaded.name, uploaded.size, model_key),
        "label": label,
        "score": score,
        "threshold": cfg["threshold"],
    }

res = st.session_state.get("result")
if res and res["sig"] == (uploaded.name, uploaded.size, model_key):
    c1, c2 = st.columns(2)
    c1.metric("Classe prédite", res["label"])
    c2.metric("Score PNEUMONIA", f"{res['score']:.2%}")
    st.progress(min(max(res["score"], 0.0), 1.0))
    st.caption(
        f"Score = sortie sigmoïde du modèle (P(PNEUMONIA), non calibrée). "
        f"Seuil de décision : {res['threshold']}."
    )

st.divider()
st.caption("⚠️ Prototype pédagogique – ne pas utiliser pour un diagnostic médical.")