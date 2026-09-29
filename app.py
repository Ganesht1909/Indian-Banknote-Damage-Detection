from pathlib import Path
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Indian Banknote Damage Assessment", page_icon="💵")

st.title("💵 Indian Banknote Damage Assessment")
st.caption("Portfolio demo — model predictions are not an RBI or bank decision.")

MODEL_PATH = Path("models/damage_severity_model.keras")

RECOMMENDATIONS = {
    "Good": "No damage detected by the model. Check the physical note if uncertain.",
    "Slightly Worn": "Minor wear detected. For exchange questions, refer to current RBI/bank guidance.",
    "Torn": "Damage detected. A bank branch can assess a soiled/mutilated/imperfect note under RBI rules.",
    "Dirty / Soiled": "Soiling detected. RBI guidance provides for exchange of soiled notes at bank branches.",
    "Crumpled": "Visible wear/damage detected. A bank branch can provide an official assessment.",
    "Missing Corner": "Possible mutilation detected. A bank branch should assess exchange value under applicable rules.",
    "Heavily Damaged": "Severe damage detected. Seek an official bank/RBI assessment; this app does not determine refund value."
}

uploaded = st.file_uploader("Upload a banknote image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image")

    if not MODEL_PATH.exists():
        st.warning(
            "No trained damage-severity model is included yet. "
            "Train 03_Damage_Severity_Classification.ipynb first."
        )
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = [
            "Good",
            "Slightly Worn",
            "Torn",
            "Dirty / Soiled",
            "Crumpled",
            "Missing Corner",
            "Heavily Damaged"
        ]

        x = image.resize((224,224))
        x = np.asarray(x, dtype=np.float32)
        x = np.expand_dims(x, 0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

        pred = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        label = class_names[idx]
        confidence = float(pred[idx])

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.1%}")
        st.info(RECOMMENDATIONS[label])

st.markdown("---")
st.caption(
    "This is an educational portfolio application. "
    "Official exchange/refund decisions are made under RBI rules and by authorised bank/RBI processes."
)
