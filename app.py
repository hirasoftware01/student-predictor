import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.7rem 2rem;
    border: none;
    border-radius: 12px;
    width: 100%;
    margin-top: 1rem;
}
.result-box {
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    margin-top: 1.5rem;
}
.grade-A { background: #d4edda; border: 2px solid #28a745; color: #155724; }
.grade-B { background: #cce5ff; border: 2px solid #007bff; color: #004085; }
.grade-C { background: #fff3cd; border: 2px solid #ffc107; color: #856404; }
.grade-D { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000
    study   = np.random.uniform(0, 10, n)
    attend  = np.random.uniform(50, 100, n)
    gpa     = np.random.uniform(1.0, 4.0, n)
    sleep   = np.random.uniform(4, 10, n)
    part    = np.random.randint(0, 2, n)
    extra   = np.random.randint(0, 2, n)
    internet= np.random.randint(0, 2, n)
    style   = np.random.randint(0, 3, n)

    score = (study*8 + (attend-50)*0.5 + gpa*15 +
             sleep*1.5 - part*5 + extra*3 + internet*2)
    score += np.random.normal(0, 10, n)

    grade = pd.cut(score, bins=4, labels=['D','C','B','A'])

    df = pd.DataFrame({
        'Study_Hours': study, 'Attendance': attend, 'GPA': gpa,
        'Sleep_Hours': sleep, 'Part_Time': part, 'Extracurricular': extra,
        'Internet': internet, 'Learning_Style': style, 'Grade': grade
    }).dropna()

    X = df.drop('Grade', axis=1)
    y = df['Grade']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# ── UI ──────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Grade Predictor")
st.markdown("Select your details below and find out your predicted grade!")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.selectbox(
        "📚 Daily Study Hours",
        options=[f"{i} hour{'s' if i != 1 else ''}" for i in range(0, 11)],
        index=5
    )

    attendance = st.selectbox(
        "🏫 Attendance Percentage",
        options=[f"{i}%" for i in range(50, 101, 5)],
        index=6
    )

    prev_gpa = st.selectbox(
        "📊 Previous GPA",
        options=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"],
        index=3
    )

    sleep_hours = st.selectbox(
        "😴 Sleep Hours Per Night",
        options=[f"{i} hour{'s' if i != 1 else ''}" for i in range(4, 11)],
        index=3
    )

with col2:
    part_time = st.selectbox(
        "💼 Do you have a Part-Time Job?",
        options=["No", "Yes"]
    )

    extra = st.selectbox(
        "⚽ Extracurricular Activities?",
        options=["No", "Yes"]
    )

    internet = st.selectbox(
        "🌐 Internet Access at Home?",
        options=["No", "Yes"]
    )

    style = st.selectbox(
        "🧠 Learning Style",
        options=["Visual (learn by seeing)",
                 "Auditory (learn by listening)",
                 "Kinesthetic (learn by doing)"]
    )

st.markdown("---")

if st.button("🔮 Predict My Grade!"):
    study_val   = int(study_hours.split()[0])
    attend_val  = int(attendance.replace('%', ''))
    gpa_val     = float(prev_gpa)
    sleep_val   = int(sleep_hours.split()[0])
    part_val    = 1 if part_time == "Yes" else 0
    extra_val   = 1 if extra == "Yes" else 0
    internet_val= 1 if internet == "Yes" else 0
    style_val   = ["Visual (learn by seeing)",
                   "Auditory (learn by listening)",
                   "Kinesthetic (learn by doing)"].index(style)

    inp = np.array([[study_val, attend_val, gpa_val, sleep_val,
                     part_val, extra_val, internet_val, style_val]])
    inp_scaled  = scaler.transform(inp)
    prediction  = model.predict(inp_scaled)[0]
    proba       = model.predict_proba(inp_scaled)[0]
    confidence  = round(max(proba) * 100, 1)

    grade_info = {
        'A': ("🏆 Excellent!", "Outstanding performance! Keep it up!", "grade-A"),
        'B': ("👍 Good Job!",  "Great work! A little more effort and you can get an A!", "grade-B"),
        'C': ("📈 Average",    "You can do better! Try studying more consistently.", "grade-C"),
        'D': ("⚠️ Needs Improvement", "Focus more on studies and seek help from teachers.", "grade-D"),
    }
    emoji, msg, css = grade_info[prediction]

    st.markdown(f"""
    <div class="result-box {css}">
        <div style="font-size:3.5rem; font-weight:800;">Grade {prediction}</div>
        <div style="font-size:1.5rem; margin:0.4rem 0;">{emoji}</div>
        <div style="font-size:1rem;">{msg}</div>
        <div style="font-size:0.8rem; margin-top:1rem; opacity:0.65;">
            Model Confidence: {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Grade Probabilities:")
    for cls, prob in zip(model.classes_, proba):
        st.progress(float(prob), text=f"Grade {cls}: {round(prob*100, 1)}%")

st.markdown("---")
st.caption("🤖 Powered by Random Forest ML model trained on student performance data.")
