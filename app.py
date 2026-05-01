import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Predictor", layout="wide")

st.title("🎓 Student Performance Predictor")

# Load dataset
df = pd.read_csv("student-mat.csv", sep=';')

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features
X = df[['studytime', 'failures', 'absences', 'health', 'freetime']]
y = df['G3']

# Train model (🔥 done inside app)
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Student Details")

studytime = st.slider("Study Time (1–4)", 1, 4)
failures = st.slider("Failures (0–3)", 0, 3)
absences = st.slider("Absences", 0, 50)
health = st.slider("Health (1–5)", 1, 5)
freetime = st.slider("Free Time (1–5)", 1, 5)

if st.button("Predict"):
    input_data = np.array([[studytime, failures, absences, health, freetime]])
    prediction = model.predict(input_data)[0]

    st.success(f"📊 Predicted Marks: {prediction:.2f}")

    if prediction >= 75:
        st.success("Excellent 🚀")
    elif prediction >= 50:
        st.warning("Average ⚠️")
    else:
        st.error("Needs Improvement ❗")
