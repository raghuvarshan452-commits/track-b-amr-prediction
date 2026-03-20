import os
os.chdir(r'C:\Users\raghu\track_b_amr')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from pathlib import Path

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Predictor — CodeCure",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Dark clinical biotech aesthetic ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --teal:    #00D4AA;
    --teal2:   #00A888;
    --navy:    #040D1A;
    --navy2:   #071428;
    --navy3:   #0A1F3A;
    --card:    #0D2040;
    --border:  rgba(0,212,170,0.2);
    --red:     #FF4E6A;
    --amber:   #FFB627;
    --green:   #00D4AA;
    --text:    #E8F4F8;
    --muted:   #7A9BB5;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

.stApp {
    background: linear-gradient(135deg, #040D1A 0%, #071428 50%, #040D1A 100%) !important;
}

/* Animated background grid */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0D2040 0%, #071428 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    color: var(--teal);
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--teal), var(--teal2));
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--teal);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Input section */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stSelectbox label {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal2) 100%) !important;
    color: var(--navy) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
    min-width: 320px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,170,0.3) !important;
}

/* Result cards */
.result-resistant {
    background: linear-gradient(135deg, rgba(255,78,106,0.15) 0%, rgba(255,78,106,0.05) 100%);
    border: 1px solid rgba(255,78,106,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}

.result-susceptible {
    background: linear-gradient(135deg, rgba(0,212,170,0.15) 0%, rgba(0,212,170,0.05) 100%);
    border: 1px solid rgba(0,212,170,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}

.result-intermediate {
    background: linear-gradient(135deg, rgba(255,182,39,0.15) 0%, rgba(255,182,39,0.05) 100%);
    border: 1px solid rgba(255,182,39,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-amber 2s infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,78,106,0); }
    50% { box-shadow: 0 0 20px 4px rgba(255,78,106,0.15); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,170,0); }
    50% { box-shadow: 0 0 20px 4px rgba(0,212,170,0.15); }
}
@keyframes pulse-amber {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,182,39,0); }
    50% { box-shadow: 0 0 20px 4px rgba(255,182,39,0.15); }
}

.result-class {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.result-confidence {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Feature importance cards */
.feature-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.feature-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--teal);
    width: 24px;
}

.feature-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
}

.feature-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--teal), var(--teal2));
    border-radius: 3px;
    transition: width 1s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Info boxes */
.stAlert {
    background: var(--card) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--teal2); border-radius: 3px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, le, feature_cols

model, le, feature_cols = load_model()


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem;
                    color: #00D4AA; font-weight: 700; letter-spacing: -0.5px;'>
            AMR<span style='color:#7A9BB5;'>predictor</span>
        </div>
        <div style='font-size: 0.7rem; color: #7A9BB5; letter-spacing: 2px;
                    text-transform: uppercase; margin-top: 0.3rem;'>
            CodeCure · Track B
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Model Stats</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card' style='margin-bottom:0.8rem;'>
        <div class='metric-val'>0.735</div>
        <div class='metric-label'>CV F1 Score</div>
    </div>
    <div class='metric-card' style='margin-bottom:0.8rem;'>
        <div class='metric-val'>60%</div>
        <div class='metric-label'>Test Accuracy</div>
    </div>
    <div class='metric-card' style='margin-bottom:0.8rem;'>
        <div class='metric-val'>274</div>
        <div class='metric-label'>Real Isolates</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.82rem; color: #7A9BB5; line-height: 1.7;'>
    Predicts <span style='color:#00D4AA;'>Gentamicin resistance</span>
    in bacterial isolates using Random Forest trained on
    274 real environmental isolates from Nigeria.
    <br><br>
    Features: zone of inhibition patterns + collection geography.
    <br><br>
    <span style='color:#7A9BB5;'>SHAP explainability built-in.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color: #3A5A70; text-align:center; line-height:1.8;'>
    Random Forest · SMOTE balanced<br>
    Mendeley AMR Dataset · CLSI breakpoints<br>
    CodeCure AI Hackathon 2026
    </div>
    """, unsafe_allow_html=True)


# ── Hero Header ────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='hero-badge'>🧬 TRACK B — ANTIMICROBIAL RESISTANCE</div>
    <div class='hero-title'>Antibiotic Resistance<br>Prediction System</div>
    <div class='hero-subtitle'>
        Enter a bacterial isolate's known resistance profile to predict
        Gentamicin susceptibility using ML + SHAP explainability
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input Section ──────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='section-header'>📍 Collection Profile</div>",
                unsafe_allow_html=True)
    city = st.selectbox(
        "Collection city",
        ['IFE', 'OSU', 'IWO', 'EDE'],
        help="City where bacterial isolate was collected in Nigeria"
    )
    source = st.selectbox(
        "Collection source",
        ['Butcher_Table', 'Concrete_Slab', 'Surrounding_Soil'],
        format_func=lambda x: x.replace('_', ' '),
        help="Environmental source of sample collection"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: rgba(0,212,170,0.06); border: 1px solid rgba(0,212,170,0.15);
                border-radius: 10px; padding: 1rem; font-size: 0.8rem; color: #7A9BB5;
                line-height: 1.7;'>
    <span style='color:#00D4AA; font-weight:600;'>Note:</span>
    Values are converted from zone of inhibition (mm) measurements
    using CLSI/EUCAST breakpoints. 0mm = fully Resistant.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='section-header'>💊 Known Resistance Profile</div>",
                unsafe_allow_html=True)

    options     = ['Susceptible', 'Intermediate', 'Resistant']
    imipenem    = st.selectbox("Imipenem",      options, key='imp')
    ceftazidime = st.selectbox("Ceftazidime",   options, key='cef')
    augmentin   = st.selectbox("Augmentin",     options, key='aug')
    cipro       = st.selectbox("Ciprofloxacin", options, key='cip')

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────
predict = st.button("⚡ Analyse Resistance Profile", type="primary")

st.markdown("<hr>", unsafe_allow_html=True)


# ── Prediction Logic ───────────────────────────────────────
if predict:
    input_data = {
        'City': city, 'Source': source,
        'IMIPENEM': imipenem, 'CEFTAZIDIME': ceftazidime,
        'AUGMENTIN': augmentin, 'CIPROFLOXACIN': cipro,
    }
    input_df      = pd.DataFrame([input_data])
    cats          = ['City', 'Source', 'IMIPENEM', 'CEFTAZIDIME', 'AUGMENTIN', 'CIPROFLOXACIN']
    input_encoded = pd.get_dummies(input_df, columns=cats)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded    = input_encoded[feature_cols].astype(float)
    prediction       = model.predict(input_encoded)[0]
    probabilities    = model.predict_proba(input_encoded)[0]
    predicted_class  = le.inverse_transform([prediction])[0]
    confidence       = max(probabilities) * 100

    # ── Result display ──────────────────────────────────────
    color_map  = {'Resistant': '#FF4E6A', 'Intermediate': '#FFB627', 'Susceptible': '#00D4AA'}
    class_map  = {'Resistant': 'result-resistant',
                  'Intermediate': 'result-intermediate',
                  'Susceptible': 'result-susceptible'}
    icon_map   = {'Resistant': '⚠️', 'Intermediate': '⚡', 'Susceptible': '✅'}
    risk_map   = {'Resistant': 'HIGH RISK', 'Intermediate': 'MEDIUM RISK', 'Susceptible': 'LOW RISK'}

    res1, res2, res3 = st.columns(3)

    with res1:
        st.markdown(f"""
        <div class='{class_map[predicted_class]}'>
            <div style='font-size:0.75rem; color:#7A9BB5; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:0.5rem;'>Prediction</div>
            <div style='font-size:2rem;'>{icon_map[predicted_class]}</div>
            <div class='result-class' style='color:{color_map[predicted_class]};
                        font-size:1.4rem; white-space:nowrap;'>{predicted_class}</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div class='{class_map[predicted_class]}'>
            <div style='font-size:0.75rem; color:#7A9BB5; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:0.5rem;'>Confidence</div>
            <div style='font-size:2rem;'>🎯</div>
            <div class='result-class' style='color:{color_map[predicted_class]};
                        font-size:1.4rem; white-space:nowrap;'>{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with res3:
        st.markdown(f"""
        <div class='{class_map[predicted_class]}'>
            <div style='font-size:0.75rem; color:#7A9BB5; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:0.5rem;'>Clinical Risk</div>
            <div style='font-size:2rem;'>🏥</div>
            <div class='result-class' style='color:{color_map[predicted_class]};
                        font-size:1.2rem; white-space:nowrap;'>{risk_map[predicted_class]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two column layout for chart + interpretation ────────
    chart_col, interp_col = st.columns([1.2, 1], gap="large")

    with chart_col:
        st.markdown("<div class='section-header'>📊 Probability Distribution</div>",
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#0D2040')
        ax.set_facecolor('#0D2040')

        classes = le.classes_
        colors  = ['#FF4E6A' if c == 'Resistant'
                   else '#FFB627' if c == 'Intermediate'
                   else '#00D4AA' for c in classes]
        probs   = [probabilities[list(le.classes_).index(c)] * 100 for c in classes]

        bars = ax.barh(classes, probs, color=colors,
                       edgecolor='none', height=0.5)

        for bar, val in zip(bars, probs):
            ax.text(min(val + 1, 95), bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=12,
                    fontweight='bold', color='white')

        ax.set_xlim(0, 110)
        ax.set_xlabel('Probability (%)', fontsize=10, color='#7A9BB5')
        ax.tick_params(colors='#7A9BB5', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1A3A5C')
        ax.spines['left'].set_color('#1A3A5C')
        ax.xaxis.label.set_color('#7A9BB5')

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        ax.grid(axis='x', color='#00D4AA', alpha=0.08,
                linewidth=0.5, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with interp_col:
        st.markdown("<div class='section-header'>🏥 Clinical Guidance</div>",
                    unsafe_allow_html=True)

        if predicted_class == 'Resistant':
            st.markdown("""
            <div style='background:rgba(255,78,106,0.1); border:1px solid rgba(255,78,106,0.3);
                        border-radius:12px; padding:1.2rem;'>
                <div style='color:#FF4E6A; font-weight:600; font-size:0.9rem;
                            margin-bottom:0.8rem;'>⚠️ HIGH RISK — Do Not Use Gentamicin</div>
                <div style='color:#B0C8D8; font-size:0.82rem; line-height:1.8;'>
                • Gentamicin treatment NOT recommended<br>
                • Confirmatory susceptibility testing required<br>
                • Consider alternative aminoglycosides or non-aminoglycoside therapy<br>
                • Check for co-resistance with other antibiotics
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_class == 'Intermediate':
            st.markdown("""
            <div style='background:rgba(255,182,39,0.1); border:1px solid rgba(255,182,39,0.3);
                        border-radius:12px; padding:1.2rem;'>
                <div style='color:#FFB627; font-weight:600; font-size:0.9rem;
                            margin-bottom:0.8rem;'>⚡ MEDIUM RISK — Use With Caution</div>
                <div style='color:#B0C8D8; font-size:0.82rem; line-height:1.8;'>
                • Treatment possible with increased dosage<br>
                • Confirmatory testing strongly recommended<br>
                • Monitor patient response closely<br>
                • Consider MIC determination before treatment
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(0,212,170,0.1); border:1px solid rgba(0,212,170,0.3);
                        border-radius:12px; padding:1.2rem;'>
                <div style='color:#00D4AA; font-weight:600; font-size:0.9rem;
                            margin-bottom:0.8rem;'>✅ LOW RISK — Gentamicin Likely Effective</div>
                <div style='color:#B0C8D8; font-size:0.82rem; line-height:1.8;'>
                • Standard Gentamicin dosing protocol applicable<br>
                • Treatment likely to be effective<br>
                • Routine monitoring recommended<br>
                • Re-test if no clinical improvement in 48–72h
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Feature Importance ─────────────────────────────
    st.markdown("<div class='section-header'>🔍 Key Predictors — SHAP Explainability</div>",
                unsafe_allow_html=True)

    explainer    = shap.TreeExplainer(model)
    shap_vals    = explainer.shap_values(input_encoded)
    arr          = np.array(shap_vals)
    resistant_idx = list(le.classes_).index('Resistant')

    if len(arr.shape) == 3:
        sv = arr[:, :, resistant_idx]
    else:
        sv = shap_vals[resistant_idx]

    shap_series = pd.Series(
        sv[0], index=feature_cols
    ).abs().sort_values(ascending=False).head(5)

    max_val = shap_series.max()
    feat_cols = st.columns(5)

    for i, (feat, val) in enumerate(shap_series.items()):
        pct   = int((val / max_val) * 100)
        clean = feat.replace('_', ' ')
        parts = clean.split(' ')
        drug  = parts[0] if len(parts) > 0 else clean
        status = parts[1] if len(parts) > 1 else ''

        with feat_cols[i]:
            st.markdown(f"""
            <div style='background:#0D2040; border:1px solid rgba(0,212,170,0.2);
                        border-radius:10px; padding:1rem; text-align:center;
                        height:130px; display:flex; flex-direction:column;
                        justify-content:center; align-items:center;'>
                <div style='font-size:0.65rem; color:#7A9BB5; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:0.4rem;'>#{i+1}</div>
                <div style='font-size:0.8rem; color:#00D4AA; font-weight:600;
                            margin-bottom:0.2rem;'>{drug}</div>
                <div style='font-size:0.7rem; color:#B0C8D8; margin-bottom:0.6rem;'>{status}</div>
                <div style='width:100%; background:rgba(255,255,255,0.05);
                            height:4px; border-radius:2px; overflow:hidden;'>
                    <div style='width:{pct}%; height:100%;
                                background:linear-gradient(90deg,#00D4AA,#00A888);
                                border-radius:2px;'></div>
                </div>
                <div style='font-size:0.7rem; color:#7A9BB5; margin-top:0.4rem;'>
                    {val:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.1);
                border-radius:10px; padding:1rem 1.5rem; font-size:0.8rem;
                color:#7A9BB5; line-height:1.8;'>
    <span style='color:#00D4AA; font-weight:600;'>How to read SHAP scores:</span>
    Higher values = stronger influence on this prediction.
    These scores are unique to THIS specific isolate — different input combinations
    will produce different explanations.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state ─────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;
                border: 1px dashed rgba(0,212,170,0.2);
                border-radius: 16px; background: rgba(0,212,170,0.02);'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🔬</div>
        <div style='font-family: Space Mono, monospace; font-size: 1rem;
                    color: #00D4AA; margin-bottom: 0.5rem;'>
            Ready to analyse
        </div>
        <div style='font-size: 0.85rem; color: #7A9BB5; line-height: 1.7;'>
            Select the collection profile and known resistance results above,<br>
            then click <strong style='color:#00D4AA;'>Analyse Resistance Profile</strong>
            to get a prediction with SHAP explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)