import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from train_model import train_and_save_model

# Page config
st.set_page_config(
    page_title="Tech Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# Train model if not already trained
if not os.path.exists("model.pkl"):
    train_and_save_model()

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ---------- UI ----------
st.title("💰 Tech Salary Predictor")
st.markdown("Predict your expected salary based on your profile — built with Machine Learning!")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("🧑‍💻 Job Title", [
        "Data Scientist", "Data Analyst", "ML Engineer",
        "Software Engineer", "Data Engineer", "AI Researcher"
    ])

    experience = st.slider("📅 Years of Experience", 0, 20, 2)

    education = st.selectbox("🎓 Education Level", [
        "Bachelor's", "Master's", "PhD"
    ])

with col2:
    location = st.selectbox("📍 Location", [
        "Bangalore", "Mumbai", "Hyderabad",
        "Delhi", "Pune", "Chennai", "Remote"
    ])

    skills = st.multiselect("🛠 Skills", [
        "Python", "SQL", "Machine Learning", "Deep Learning",
        "TensorFlow", "PowerBI", "Tableau", "Excel",
        "NLP", "Computer Vision", "Spark", "AWS"
    ], default=["Python", "SQL"])

    company_size = st.selectbox("🏢 Company Size", [
        "Startup (< 50)", "Mid-size (50-500)", "Large (500+)"
    ])

st.markdown("---")

if st.button("🔮 Predict My Salary", use_container_width=True):
    # Encode inputs
    job_enc = encoders["job"].transform([job_title])[0]
    edu_enc = encoders["edu"].transform([education])[0]
    loc_enc = encoders["loc"].transform([location])[0]
    size_enc = encoders["size"].transform([company_size])[0]
    skill_count = len(skills)

    # Predict
    input_data = np.array([[job_enc, experience, edu_enc, loc_enc, skill_count, size_enc]])
    predicted = model.predict(input_data)[0]

    # Display result
    st.success(f"### 🎯 Estimated Annual Salary: ₹{predicted:,.0f}")

    # Breakdown
    st.markdown("#### 📊 Salary Breakdown")
    col3, col4, col5 = st.columns(3)
    col3.metric("Monthly", f"₹{predicted/12:,.0f}")
    col4.metric("Annual", f"₹{predicted:,.0f}")
    col5.metric("Experience Bonus", f"₹{experience * 50000:,.0f}")

    st.info("💡 Tip: Adding more in-demand skills like Deep Learning or AWS can increase your salary by 15-25%!")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built by Shalini Jaiswal · "
    "<a href='https://github.com/cyber-boop1'>GitHub</a> · "
    "<a href='https://linkedin.com/in/jaiswal-shalini'>LinkedIn</a></div>",
    unsafe_allow_html=True
)
