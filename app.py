import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.main { background: #f8f9ff; }
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
    font-size: 1.1rem;
}
.grade-A { background: #d4edda; border: 2px solid #28a745; color: #155724; }
.grade-B { background: #cce5ff; border: 2px solid #007bff; color: #004085; }
.grade-C { background: #fff3cd; border: 2px solid #ffc107; color: #856404; }
.grade-D { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ── Train model on synthetic data that mirrors the real dataset ──
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000

    study   = np.random.uniform(0, 10, n)
    attend  = np.random.uniform(50, 100, n)
    gpa     = np.random.uniform(1.0, 4.0, n)
    sleep   = np.random.uniform(4, 10, n)
    part    = np.random.randint(0, 2, n)          # part-time job: 0/1
    extra   = np.random.randint(0, 2, n)          # extracurricular: 0/1
    internet= np.random.randint(0, 2, n)
    style   = np.random.randint(0, 3, n)          # Visual/Auditory/Kinesthetic

    score = (study*8 + (attend-50)*0.5 + gpa*15 +
             sleep*1.5 - part*5 + extra*3 + internet*2)
    noise = np.random.normal(0, 10, n)
    score += noise

    grade = pd.cut(score, bins=4, labels=['D','C','B','A'])

    df = pd.DataFrame({
        'Study_Hours_Per_Day': study,
        'Attendance_Rate': attend,
        'Previous_GPA': gpa,
        'Sleep_Hours_Per_Night': sleep,
        'Part_Time_Job': part,
        'Extracurricular_Activities': extra,
        'Internet_Access_at_Home': internet,
        'Learning_Style': style,
        'Grade': grade
    })
    df.dropna(inplace=True)

    X = df.drop('Grade', axis=1)
    y = df['Grade']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()

# ── UI ──────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Grade Predictor")
st.markdown("**Apna data enter karo — hum batayenge tumhara grade kya hoga!**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Rozana kitne ghante padhte ho?", 0.0, 10.0, 5.0, 0.5)
    attendance  = st.slider("🏫 Attendance (%) kitni hai?", 50, 100, 80)
    prev_gpa    = st.slider("📊 Pichla GPA kya tha?", 1.0, 4.0, 2.5, 0.1)
    sleep_hours = st.slider("😴 Raat ko kitne ghante sote ho?", 4.0, 10.0, 7.0, 0.5)

with col2:
    part_time   = st.radio("💼 Part-time job karte ho?", ["Nahi", "Haan"])
    extra       = st.radio("⚽ Extracurricular activities mein hisse lete ho?", ["Nahi", "Haan"])
    internet    = st.radio("🌐 Ghar mein internet hai?", ["Nahi", "Haan"])
    style       = st.selectbox("🧠 Tumhara learning style kya hai?", 
                               ["Visual (dekhke seekhna)", 
                                "Auditory (sunke seekhna)", 
                                "Kinesthetic (karke seekhna)"])

style_map = {"Visual (dekhke seekhna)": 0, 
             "Auditory (sunke seekhna)": 1, 
             "Kinesthetic (karke seekhna)": 2}

if st.button("🔮 Mera Grade Batao!"):
    inp = np.array([[
        study_hours,
        attendance,
        prev_gpa,
        sleep_hours,
        1 if part_time == "Haan" else 0,
        1 if extra == "Haan" else 0,
        1 if internet == "Haan" else 0,
        style_map[style]
    ]])
    inp_scaled = scaler.transform(inp)
    prediction = model.predict(inp_scaled)[0]
    proba      = model.predict_proba(inp_scaled)[0]
    confidence = round(max(proba) * 100, 1)

    grade_info = {
        'A': ("🏆 Excellent!", "Tumhara performance bohat acha hai! Keep it up!", "grade-A"),
        'B': ("👍 Good Job!", "Tumhara grade acha hai, thodi aur mehnat se A aa sakta hai!", "grade-B"),
        'C': ("📈 Average", "Thodi zyada mehnat karo, tum behtar kar sakte ho!", "grade-C"),
        'D': ("⚠️ Needs Improvement", "Zyada dhyan do padhne par — teachers se madad lo!", "grade-D"),
    }
    emoji, msg, css = grade_info[prediction]

    st.markdown(f"""
    <div class="result-box {css}">
        <div style="font-size:3rem; font-weight:700;">Grade {prediction}</div>
        <div style="font-size:1.5rem; margin: 0.5rem 0;">{emoji}</div>
        <div style="font-size:1rem;">{msg}</div>
        <div style="font-size:0.85rem; margin-top:1rem; opacity:0.7;">
            Model Confidence: {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Tumhare saare grades ki probability:")
    classes = model.classes_
    for cls, prob in zip(classes, proba):
        st.progress(float(prob), text=f"Grade {cls}: {round(prob*100, 1)}%")

st.markdown("---")
st.caption("🤖 Ye prediction Random Forest model se hai jo student data par train hua hai.")
