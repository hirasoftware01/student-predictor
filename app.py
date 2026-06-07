import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Learning Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.main { background: #0d0d0d; }
section[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #222; }

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-card .label { color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-card .value { color: #f0f0f0; font-size: 2rem; font-family: 'Space Mono', monospace; font-weight: 700; }
.metric-card .sub   { color: #555; font-size: 0.8rem; margin-top: 4px; }

.best-badge {
    display: inline-block;
    background: #39ff14;
    color: #000;
    font-weight: 700;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    margin-left: 8px;
}

.section-header {
    border-left: 4px solid #39ff14;
    padding-left: 14px;
    margin: 2rem 0 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #e0e0e0;
}

.stButton > button {
    background: #39ff14 !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

.stProgress > div > div { background: #39ff14 !important; }

div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Student Performance & Learning Style\n*(Kaggle — adilshamim8)*")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("- Logistic Regression\n- Decision Tree\n- SVM (RBF kernel)\n- Random Forest")
    st.markdown("---")
    test_size = st.slider("Test Split %", 10, 40, 20, step=5)
    random_state = st.number_input("Random Seed", value=42, step=1)
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)
    st.markdown("---")
    st.caption("Built with Streamlit · sklearn")

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Predicting Student Learning")
st.markdown("*End-to-end ML pipeline — data → preprocessing → training → evaluation*")
st.markdown("---")

# ─── Helpers ───────────────────────────────────────────────────────────────────
DARK_BG    = "#0d0d0d"
CARD_BG    = "#1a1a1a"
ACCENT     = "#39ff14"
TEXT_MAIN  = "#e0e0e0"
TEXT_DIM   = "#666"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  TEXT_MAIN,
    "xtick.color":      TEXT_DIM,
    "ytick.color":      TEXT_DIM,
    "text.color":       TEXT_MAIN,
    "grid.color":       "#222",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

@st.cache_data(show_spinner=False)
def load_and_preprocess(test_pct, seed):
    try:
        import kagglehub
        path = kagglehub.dataset_download("adilshamim8/student-performance-and-learning-style")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        source = "kaggle"
    except Exception:
        # Fallback synthetic dataset
        np.random.seed(seed)
        n = 1000
        df = pd.DataFrame({
            "study_hours": np.random.uniform(0, 10, n),
            "attendance":  np.random.uniform(50, 100, n),
            "prev_gpa":    np.random.uniform(1.0, 4.0, n),
            "sleep_hours": np.random.uniform(4, 10, n),
            "learning_style": np.random.choice(["Visual","Auditory","Kinesthetic"], n),
            "Grade_Category": np.random.choice(["A","B","C","D"], n, p=[0.25,0.35,0.25,0.15]),
        })
        source = "synthetic"

    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64','int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    grade_col = [c for c in df.columns if 'grade' in c.lower() or 'category' in c.lower()][0]

    num_cols = [c for c in df.select_dtypes(include='number').columns if c != grade_col]
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]

    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    target_le = le_dict.get(grade_col)

    y = df[grade_col]
    X = df.drop(columns=[grade_col])
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_pct/100, random_state=seed, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test, target_le, grade_col, source

@st.cache_data(show_spinner=False)
def train_models(seed, _X_train, _y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=seed),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=seed),
        'SVM':                 SVC(kernel='rbf', C=5.0, gamma='scale', random_state=seed),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=seed),
    }
    trained, cv_scores = {}, {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        trained[name] = model
        cv = cross_val_score(model, _X_train, _y_train, cv=5, scoring='accuracy')
        cv_scores[name] = (cv.mean(), cv.std())
    return trained, cv_scores

def evaluate(trained, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'train_acc': accuracy_score(y_train, model.predict(X_train)),
            'test_acc':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1':        f1_score(y_test, y_pred, average='macro', zero_division=0),
            'cm':        confusion_matrix(y_test, y_pred),
            'y_pred':    y_pred,
            'report':    classification_report(y_test, y_pred, zero_division=0),
        }
    return results

# ─── Main execution ────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:
    with st.spinner("Loading & preprocessing data…"):
        X, y, X_train, X_test, y_train, y_test, target_le, grade_col, source = \
            load_and_preprocess(test_size, random_state)

    # ── Step 1: Data Overview ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">01 · Dataset Overview</div>', unsafe_allow_html=True)

    if source == "synthetic":
        st.warning("⚠️ Kaggle dataset unavailable — using a **synthetic fallback** dataset for demo purposes. To use real data, set up your Kaggle API credentials.")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Total Samples", f"{len(X):,}", "after deduplication & outlier removal"),
        (c2, "Features",      f"{X.shape[1]}",   "after encoding & scaling"),
        (c3, "Train / Test",  f"{len(X_train)} / {len(X_test)}", f"{100-test_size}% / {test_size}%"),
        (c4, "Classes",       f"{y.nunique()}",  f"in '{grade_col}'"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 Feature Distributions"):
        n_cols = min(4, X.shape[1])
        fig, axes = plt.subplots(1, n_cols, figsize=(16, 3))
        if n_cols == 1:
            axes = [axes]
        for ax, col_name in zip(axes, list(X.columns)[:n_cols]):
            ax.hist(X[col_name], bins=30, color=ACCENT, alpha=0.85, edgecolor='none')
            ax.set_title(col_name, fontsize=9)
            ax.set_xlabel("")
        fig.suptitle("Normalized Feature Distributions (first 4)", color=TEXT_MAIN)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("🔥 Correlation Heatmap"):
        corr = X.copy()
        corr[grade_col] = y.values
        fig, ax = plt.subplots(figsize=(10, 6))
        mask = np.triu(np.ones_like(corr.corr(), dtype=bool))
        sns.heatmap(corr.corr(), mask=mask, annot=True, fmt=".2f", cmap="Greens",
                    ax=ax, linewidths=0.3, linecolor="#111",
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
        ax.set_title("Feature Correlation Matrix", pad=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Step 2: Train ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">02 · Model Training</div>', unsafe_allow_html=True)

    with st.spinner("Training 4 models (5-fold CV)…"):
        trained, cv_scores = train_models(random_state, X_train, y_train)

    cv_col = st.columns(4)
    for col_el, (name, (mean, std)) in zip(cv_col, cv_scores.items()):
        col_el.markdown(f"""
        <div class="metric-card">
          <div class="label">{name}</div>
          <div class="value">{mean:.3f}</div>
          <div class="sub">CV Accuracy ± {std:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Step 3: Evaluate ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">03 · Evaluation Results</div>', unsafe_allow_html=True)

    results = evaluate(trained, X_train, X_test, y_train, y_test)
    best = max(results, key=lambda n: results[n]['f1'])
    st.session_state["results"] = True

    # Results table
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name + (" ⭐" if name == best else ""),
            "Train Acc": f"{res['train_acc']:.4f}",
            "Test Acc":  f"{res['test_acc']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall":    f"{res['recall']:.4f}",
            "F1 Score":  f"{res['f1']:.4f}",
            "Overfit?":  "⚠️ Yes" if (res['train_acc'] - res['test_acc']) > 0.08 else "✅ No",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.success(f"🏆 **Best Model: {best}** — F1 Score: {results[best]['f1']:.4f}")

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">04 · Visual Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Accuracy Comparison", "🔢 Confusion Matrices", "📈 CV Scores"])

    with tab1:
        names      = list(results.keys())
        train_accs = [results[n]['train_acc'] for n in names]
        test_accs  = [results[n]['test_acc']  for n in names]
        x = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars_tr = ax.bar(x - 0.22, train_accs, 0.42, label='Train', color="#333", edgecolor=ACCENT, linewidth=0.8)
        bars_te = ax.bar(x + 0.22, test_accs,  0.42, label='Test',  color=ACCENT, edgecolor=ACCENT, linewidth=0.8)
        for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
            gap = tr - te
            color = "#ff4444" if gap > 0.08 else "#39ff14"
            ax.text(i, max(tr, te) + 0.025, f'Δ{gap:+.3f}', ha='center',
                    fontsize=8, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title('Training vs Testing Accuracy', fontweight='bold', pad=12)
        ax.legend(framealpha=0.1)
        ax.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        class_names = list(target_le.classes_) if target_le else None
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for ax, (name, res) in zip(axes, results.items()):
            cm = res['cm']
            im = ax.imshow(cm, cmap='Greens', aspect='auto')
            ax.set_title(name, fontweight='bold', fontsize=9)
            if class_names:
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(class_names, fontsize=8)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=9, color='white' if cm[i,j] > cm.max()/2 else '#aaa')
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Actual', fontsize=8)
        fig.suptitle('Confusion Matrices — Test Set', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        names_cv = list(cv_scores.keys())
        means    = [cv_scores[n][0] for n in names_cv]
        stds     = [cv_scores[n][1] for n in names_cv]
        colors   = [ACCENT if n == best else "#444" for n in names_cv]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names_cv, means, yerr=stds, capsize=5, color=colors,
                      edgecolor="#555", linewidth=0.8, error_kw={"ecolor": "#888", "lw": 1.5})
        ax.set_ylim(0, 1.1)
        ax.set_title('5-Fold Cross-Validation Accuracy (mean ± std)', fontweight='bold')
        ax.grid(axis='y')
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold', color=TEXT_MAIN)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Classification Report ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">05 · Classification Report</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select model:", list(results.keys()), index=list(results.keys()).index(best))
    st.code(results[sel]['report'], language="text")

    # ── Feature Importance (RF) ────────────────────────────────────────────────
    if 'Random Forest' in trained:
        st.markdown('<div class="section-header">06 · Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        rf_model = trained['Random Forest']
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                       color=[ACCENT if i >= len(feat_df)-3 else "#333" for i in range(len(feat_df))],
                       edgecolor="none")
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importances', fontweight='bold')
        ax.grid(axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    # ── Landing state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; color: #555;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🎓</div>
        <div style="font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #888;">
            Configure settings in the sidebar<br>then click <span style="color:#39ff14;">▶ Run Pipeline</span>
        </div>
        <br>
        <div style="font-size: 0.85rem; color: #444; max-width: 520px; margin: auto;">
            This app trains Logistic Regression, Decision Tree, SVM, and Random Forest on 
            the Student Performance & Learning Style dataset, then compares them side-by-side.
        </div>
    </div>
    """, unsafe_allow_html=True)
