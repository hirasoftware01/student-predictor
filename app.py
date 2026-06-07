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
.result-excellent { background: #d4edda; border: 2px solid #28a745; color: #155724; }
.result-good      { background: #cce5ff; border: 2px solid #007bff; color: #004085; }
.result-average   { background: #fff3cd; border: 2px solid #ffc107; color: #856404; }
.result-below     { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }
.result-box {
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 4000

    study      = np.random.uniform(0, 40, n)
    attend     = np.random.uniform(50, 100, n)
    sleep      = np.random.uniform(4, 10, n)
    motivation = np.random.randint(0, 3, n)
    internet   = np.random.randint(0, 2, n)
    style      = np.random.randint(0, 4, n)
    prev_score = np.random.uniform(40, 100, n)
    extra      = np.random.randint(0, 2, n)

    score = (study * 2.5 +
             (attend - 50) * 1.5 +
             sleep * 2.0 +
             motivation * 20 +
             internet * 5 +
             prev_score * 0.8 +
             extra * 6)
    score += np.random.normal(0, 8, n)

    # 4 balanced labels using percentiles
    q25, q50, q75 = np.percentile(score, [25, 50, 75])
    def label(s):
        if s >= q75:   return 'Excellent'
        elif s >= q50: return 'Good'
        elif s >= q25: return 'Average'
        else:          return 'Below Average'

    grades = pd.Series([label(s) for s in score])

    df = pd.DataFrame({
        'Study_Hours_Per_Week'   : study,
        'Attendance_Rate'        : attend,
        'Sleep_Hours_Per_Night'  : sleep,
        'Motivation_Level'       : motivation,
        'Internet_Access'        : internet,
        'Learning_Style'         : style,
        'Previous_Academic_Score': prev_score,
        'Extracurricular'        : extra,
        'Grade'                  : grades
    })

    X = df.drop('Grade', axis=1)
    y = df['Grade']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# ── UI ──────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Grade Predictor")
st.markdown("Fill in your details below to predict your grade outcome!")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.selectbox(
        "Study Hours Per Week",
        [f"{i} hrs" for i in range(0, 41, 2)],
        index=10
    )
    attendance = st.selectbox(
        "Attendance Rate",
        [f"{i}%" for i in range(50, 101, 5)],
        index=6
    )
    sleep_hours = st.selectbox(
        "Sleep Hours Per Night",
        [f"{i} hrs" for i in range(4, 11)],
        index=3
    )
    prev_score = st.selectbox(
        "Previous Academic Score",
        [f"{i}" for i in range(40, 101, 5)],
        index=6
    )

with col2:
    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"],
        index=1
    )
    internet = st.selectbox(
        "Internet Access at Home",
        ["No", "Yes"],
        index=1
    )
    learning_style = st.selectbox(
        "Learning Style",
        ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
        index=0
    )
    extra = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"],
        index=0
    )

st.markdown("---")

if st.button("🔮 Predict My Grade!"):
    study_val  = int(study_hours.split()[0])
    attend_val = int(attendance.replace('%', ''))
    sleep_val  = int(sleep_hours.split()[0])
    prev_val   = int(prev_score)
    motiv_val  = ["Low", "Medium", "High"].index(motivation)
    net_val    = 1 if internet == "Yes" else 0
    style_val  = ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"].index(learning_style)
    extra_val  = 1 if extra == "Yes" else 0

    inp = np.array([[study_val, attend_val, sleep_val,
                     motiv_val, net_val, style_val,
                     prev_val, extra_val]])
    inp_scaled = scaler.transform(inp)
    prediction = model.predict(inp_scaled)[0]
    proba      = model.predict_proba(inp_scaled)[0]
    confidence = round(max(proba) * 100, 1)

    grade_info = {
        'Excellent'    : ("🏆 Excellent!", "Outstanding! You are performing at the highest level.", "result-excellent"),
        'Good'         : ("👍 Good!",       "Great job! You are performing well above average.",    "result-good"),
        'Average'      : ("📈 Average",     "You are doing okay, but there is room to improve.",   "result-average"),
        'Below Average': ("⚠️ Below Average","You need to work harder to improve your performance.", "result-below"),
    }
    emoji, msg, css = grade_info[prediction]

    st.markdown(f"""
    <div class="result-box {css}">
        <div style="font-size:3rem; font-weight:800;">{prediction}</div>
        <div style="font-size:1.5rem; margin:0.4rem 0;">{emoji}</div>
        <div style="font-size:1rem;">{msg}</div>
        <div style="font-size:0.8rem; margin-top:1rem; opacity:0.65;">
            Model Confidence: {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Prediction Probabilities:")
    for cls, prob in zip(model.classes_, proba):
        st.progress(float(prob), text=f"{cls}: {round(prob*100, 1)}%")

st.markdown("---")
st.caption("🤖 Powered by Random Forest ML model | Features match Assignment 3 dataset")
