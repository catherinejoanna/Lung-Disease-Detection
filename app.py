# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Integrated Lung Disease Detection",
    layout="wide"
)

st.title("🫁 Integrated Lung Disease Detection System")
st.caption("Asthma • Chest Cancer • Pneumonia • Tuberculosis")
st.markdown("---")

device = torch.device("cpu")

# ================= MODEL LOADERS =================
@st.cache_resource
def load_resnet_model(model_path, num_classes):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

@st.cache_resource
def load_models():
    cancer_model = load_resnet_model(
        "G:\Integratedpth\best_lung_cancer_model.pth", 4
    )

    pneumonia_model = tf.keras.models.load_model(
        "G:\Integratedpth\pneumonia_model.h5",
        compile=False
    )

    tb_model = tf.keras.models.load_model(
        "G:/Integratedpth/tuberculosis_detector.h5",
        compile=False
    )

    asthma_model = joblib.load(
        "G:/Integratedpth/asthma_multiclass_xgb.joblib"
    )

    return cancer_model, pneumonia_model, tb_model, asthma_model

cancer_model, pneumonia_model, tb_model, asthma_model = load_models()

# ================= PREPROCESSING =================
pt_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def tf_preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= PREDICTION FUNCTIONS =================
cancer_classes = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
    "Other"
]

def predict_cancer(img):
    img = pt_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = cancer_model(img)
        probs = F.softmax(out, dim=1).numpy()[0]
    idx = np.argmax(probs)
    return cancer_classes[idx], float(probs[idx])

def predict_pneumonia(img):
    pred = pneumonia_model.predict(tf_preprocess(img), verbose=0)[0]
    return float(pred[-1])

def predict_tb(img):
    pred = tb_model.predict(tf_preprocess(img), verbose=0)[0]
    return float(pred[-1])

ASTHMA_FEATURES = [
    'Tiredness','Dry-Cough','Difficulty-in-Breathing','Sore-Throat',
    'None_Sympton','Pains','Nasal-Congestion','Runny-Nose',
    'None_Experiencing',
    'Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+',
    'Gender_Female','Gender_Male'
]

def predict_asthma(feature_dict):
    df = pd.DataFrame([feature_dict])[ASTHMA_FEATURES]
    pred = asthma_model.predict(df)[0]
    prob = asthma_model.predict_proba(df).max()
    return int(pred), float(prob)

# ================= DECISION LOGIC =================
def final_xray_decision(cancer_label, cancer_conf, pneu_conf, tb_conf):
    if tb_conf >= 0.5:
        return "High likelihood of Tuberculosis"
    elif pneu_conf >= 0.5:
        return "High likelihood of Pneumonia"
    elif cancer_label.lower() != "normal" and cancer_conf >= 0.5:
        return f"Chest cancer suspected ({cancer_label})"
    else:
        return "No significant abnormality detected"

# ================= SIDEBAR INPUT =================
st.sidebar.header("🧍 Patient Details")

age_group = st.sidebar.selectbox(
    "Age Group", ["0-9","10-19","20-24","25-59","60+"]
)
gender = st.sidebar.selectbox("Gender", ["Male","Female"])

st.sidebar.subheader("Symptoms")
symptoms = {
    "Tiredness": st.sidebar.checkbox("Tiredness"),
    "Dry-Cough": st.sidebar.checkbox("Dry Cough"),
    "Difficulty-in-Breathing": st.sidebar.checkbox("Difficulty in Breathing"),
    "Sore-Throat": st.sidebar.checkbox("Sore Throat"),
    "None_Sympton": st.sidebar.checkbox("No Symptoms"),
    "Pains": st.sidebar.checkbox("Body Pains"),
    "Nasal-Congestion": st.sidebar.checkbox("Nasal Congestion"),
    "Runny-Nose": st.sidebar.checkbox("Runny Nose"),
    "None_Experiencing": st.sidebar.checkbox("None Experiencing")
}

# Build asthma feature dict
asthma_input = {k: int(v) for k, v in symptoms.items()}

for a in ["Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+"]:
    asthma_input[a] = 1 if age_group in a else 0

asthma_input["Gender_Female"] = 1 if gender == "Female" else 0
asthma_input["Gender_Male"] = 1 if gender == "Male" else 0

# ================= XRAY UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg","png","jpeg"])

# ================= RUN INFERENCE =================
if st.button("🔍 Run Diagnosis") and uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    cancer_label, cancer_conf = predict_cancer(img)
    pneu_conf = predict_pneumonia(img)
    tb_conf = predict_tb(img)
    asthma_pred, asthma_conf = predict_asthma(asthma_input)

    asthma_map = {0:"Mild",1:"Moderate",2:"Severe"}

    # ================= RESULTS UI =================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌬 Asthma Assessment")
        st.metric("Severity", asthma_map[asthma_pred])
        st.progress(asthma_conf)
        st.caption(f"Confidence: {asthma_conf*100:.1f}%")

    with col2:
        st.markdown("### 🩻 Chest X-ray Findings")
        st.metric("Chest Cancer", cancer_label)
        st.caption(f"Confidence: {cancer_conf*100:.1f}%")
        st.progress(pneu_conf)
        st.caption(f"Pneumonia Probability: {pneu_conf*100:.1f}%")
        st.progress(tb_conf)
        st.caption(f"Tuberculosis Probability: {tb_conf*100:.1f}%")

    with col3:
        st.markdown("### 🩺 Final Impression")
        decision = final_xray_decision(
            cancer_label, cancer_conf, pneu_conf, tb_conf
        )
        if "Tuberculosis" in decision:
            st.error(decision)
        elif "Pneumonia" in decision:
            st.warning(decision)
        else:
            st.success(decision)

    st.markdown("---")
    st.image(img, caption="Uploaded Chest X-ray", width=350)