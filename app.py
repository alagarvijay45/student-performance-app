import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")
@st.cache_data
def load_and_train():
    df = pd.read_csv("student-mat.csv", sep=';')
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df[['studytime', 'failures', 'absences', 'health', 'freetime']]
    y = df['G3']
    model = LinearRegression()
    model.fit(X, y)
    return model, df
model, df = load_and_train()
col1, col2 = st.columns(2)
with col1:
    st.subheader("📥 Enter Student Details")
    studytime = st.slider("Study Time (1–4)", 1, 4)
    failures = st.slider("Failures (0–3)", 0, 3)
    absences = st.slider("Absences", 0, 50)
    health = st.slider("Health (1–5)", 1, 5)
    freetime = st.slider("Free Time (1–5)", 1, 5)
with col2:
    st.subheader("📊 Prediction Result")
    if st.button("Predict Marks"):
        input_data = np.array([[studytime, failures, absences, health, freetime]])
        prediction = model.predict(input_data)[0]
        st.metric("Predicted Marks", f"{prediction:.2f}")
        if prediction >= 75:
            st.success("🔥 Excellent Performance")
        elif prediction >= 50:
            st.warning("⚠️ Average Performance")
        else:
            st.error("❗ Needs Improvement")
        fig = plt.figure()
        plt.bar(["Predicted Marks"], [prediction])
        plt.ylim(0, 100)
        st.pyplot(fig)
st.markdown("---")
st.subheader("📈 Dataset Insights")
col3, col4 = st.columns(2)
with col3:
    st.write("Study Time vs Marks")
    fig1 = plt.figure()
    plt.scatter(df['studytime'], df['G3'])
    plt.xlabel("Study Time")
    plt.ylabel("Marks")
    st.pyplot(fig1)
with col4:
    st.write("Absences vs Marks")
    fig2 = plt.figure()
    plt.scatter(df['absences'], df['G3'])
    plt.xlabel("Absences")
    plt.ylabel("Marks")
    st.pyplot(fig2)
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made by Student | Data Science Project</p>", unsafe_allow_html=True)
