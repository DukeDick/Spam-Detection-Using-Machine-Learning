import streamlit as st
import joblib
import pandas as pd

# Load once at startup
@st.cache_resource  # keeps it in memory between reruns
def load_model():
    return joblib.load("spam_classifier.joblib")

model = load_model()

st.title("📨 SMS Spam Detector")

user_input = st.text_area("Paste an SMS message here:", height=150)

if st.button("Classify"):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0].max()
        label = "🚫 Spam" if pred == "spam" else "✅ Ham"
        st.markdown(f"### {label}  \nConfidence: **{proba:.2%}**")
    else:
        st.warning("Please enter a message.")
