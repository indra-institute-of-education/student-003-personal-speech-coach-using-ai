"""
Personal Speech Coach - AI Powered Real-Time Fluency Analyzer
Main Streamlit Application
Updated: Live recording + multi-format audio support (MP3, WAV, OGG, FLAC, M4A, etc.)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from feature_extraction import AudioFeatureExtractor
from feedback_engine import FeedbackEngine
from model_training import SpeechModelTrainer

# ── Audio conversion helper ─────────────────────────────────────────────────
def convert_to_wav(file_bytes: bytes, original_name: str) -> str | None:
    """
    Convert any audio format to WAV using pydub.
    Returns path to a temporary WAV file, or None on failure.
    """
    ext = os.path.splitext(original_name)[1].lower().lstrip(".")
    if not ext:
        ext = "wav"

    try:
        from pydub import AudioSegment

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_in_path = tmp_in.name

        audio = AudioSegment.from_file(tmp_in_path, format=ext)
        audio = audio.set_frame_rate(22050).set_channels(1)

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp_out.name, format="wav")
        os.unlink(tmp_in_path)
        return tmp_out.name

    except Exception as e:
        st.error(f"❌ Could not convert audio: {e}\n\nMake sure **ffmpeg** is installed on your system.")
        return None


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Speech Coach",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Remove top padding & hide Streamlit chrome */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { display: none !important; }

/* App dark background */
.stApp {
    background: #07090f !important;
    background-image:
        radial-gradient(ellipse at 15% 15%, rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 85%, rgba(168,85,247,0.14) 0%, transparent 55%) !important;
}

/* Main header */
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6, #818cf8);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 0.78rem;
    text-align: center;
    color: #374151;
    margin-bottom: 1.5rem;
    letter-spacing: 3.5px;
    text-transform: uppercase;
}
@keyframes shimmer {
    0%   { background-position: 0%   50%; }
    100% { background-position: 300% 50%; }
}

/* Metric card (speech metrics) */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #818cf8;
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.metric-card strong {
    color: #6b7280;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    display: block;
    margin-bottom: 0.2rem;
}
.metric-card span { color: #e2e8f0 !important; }

/* Feedback box */
.feedback-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin: 1rem 0;
}
.feedback-box h4 { color: #818cf8; margin-top: 0; }
.feedback-box p  { color: #cbd5e1; font-size: 1.05rem; line-height: 1.7; }

/* Strength / improvement boxes */
.success-box {
    background: rgba(52,211,153,0.08);
    border-left: 3px solid #34d399;
    padding: 0.65rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.35rem 0;
    color: #6ee7b7;
}
.warning-box {
    background: rgba(248,113,113,0.08);
    border-left: 3px solid #f87171;
    padding: 0.65rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.35rem 0;
    color: #fca5a5;
}

/* Live recorder box */
.recorder-box {
    background: rgba(99,102,241,0.06);
    border: 1.5px dashed rgba(129,140,248,0.5);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
    color: #e2e8f0;
}
.recorder-box h3 { color: #c4b5fd; }
.recorder-box p  { color: #94a3b8; }

/* Confidence score card */
.confidence-card {
    background: linear-gradient(135deg, rgba(129,140,248,0.15), rgba(196,181,253,0.10));
    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.confidence-label {
    color: #6b7280;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.1rem;
}
.confidence-value {
    color: #c4b5fd;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #05070d !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1rem 0.8rem !important;
}
/* Hide the radio label */
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio { margin: 0 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.5rem 0.8rem !important;
    margin: 1px 0 !important;
    border-radius: 8px !important;
    color: #4b5563 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 0.18s;
    border-left: 2px solid transparent !important;
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #c7d2fe !important;
    border-left-color: rgba(129,140,248,0.4) !important;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.22), rgba(168,85,247,0.10)) !important;
    color: #e0e7ff !important;
    border-left-color: #818cf8 !important;
}
/* Hide radio circle */
section[data-testid="stSidebar"] input[type="radio"] { display: none !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] { color: #4b5563 !important; border-radius: 8px; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.18)) !important;
    color: #e2e8f0 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    color: #fff !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(99,102,241,0.4) !important;
}

/* Dividers */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* General text */
p, li, .stMarkdown { color: #94a3b8; }
strong { color: #e2e8f0; }
h1, h2, h3, h4 { color: #e2e8f0 !important; }
label { color: #4b5563 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None

# ── Cached components ─────────────────────────────────────────────────────────
@st.cache_resource
def initialize_components():
    extractor = AudioFeatureExtractor()
    feedback = FeedbackEngine()
    return extractor, feedback

extractor, feedback_engine = initialize_components()

# ── Supported formats ─────────────────────────────────────────────────────────
SUPPORTED_FORMATS = ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'wma', 'aiff', 'webm']

# ── Run analysis helper ───────────────────────────────────────────────────────
def run_analysis(wav_path: str, display_label: str = ""):
    """Extract features → predict → display results → save history."""
    with st.spinner("🔄 Analyzing your speech…"):
        features_dict, audio_data, sr = extractor.extract_features(wav_path)

    if features_dict is None:
        st.error("❌ Error extracting features. Please check your audio file.")
        return

    # ── Waveform ──
    st.markdown("### 📊 Audio Waveform")
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#07090f')
    ax.set_facecolor('#07090f')
    time_axis = np.arange(len(audio_data)) / sr
    ax.plot(time_axis, audio_data, color='#818cf8', linewidth=0.5)
    ax.set_xlabel('Time (s)', color='#4b5563')
    ax.set_ylabel('Amplitude', color='#4b5563')
    ax.set_title('Speech Waveform', color='#e2e8f0')
    ax.tick_params(colors='#4b5563')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1f2937')
    ax.grid(True, alpha=0.15, color='#374151')
    st.pyplot(fig)
    plt.close()

    # Feature preparation
    feature_names = extractor.get_feature_names()
    feature_values = [features_dict.get(name, 0) for name in feature_names]
    feature_df = pd.DataFrame([feature_values], columns=feature_names)

    trainer = st.session_state.trainer
    features_scaled = trainer.scaler.transform(feature_df.values)
    prediction = trainer.best_model.predict(features_scaled)[0]
    prediction_proba = trainer.best_model.predict_proba(features_scaled)[0]

    fluency_label = trainer.label_encoder.inverse_transform([prediction])[0]
    confidence = max(prediction_proba) * 100

    # Score mapping
    score_map  = {"Fluent": 85, "Average": 60, "Needs Improvement": 35}
    color_map  = {"Fluent": "#34d399", "Average": "#fbbf24", "Needs Improvement": "#f87171"}
    score_value = score_map.get(fluency_label, 50)
    color       = color_map.get(fluency_label, "#818cf8")

    # ── Results header ──
    st.markdown("---")
    st.markdown("## 🎯 Analysis Results")

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        # Fluency badge
        if fluency_label == "Fluent":
            st.success(f"### ✅ {fluency_label}")
        elif fluency_label == "Average":
            st.info(f"### ⚠️ {fluency_label}")
        else:
            st.warning(f"### 📈 {fluency_label}")

        # Confidence score — shown prominently above gauge
        st.markdown(f"""
        <div class="confidence-card">
            <div class="confidence-label">Confidence Score</div>
            <div class="confidence-value">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fluency Score", 'font': {'size': 20, 'color': '#e2e8f0'}},
            delta={'reference': 70, 'increasing': {'color': '#34d399'}, 'decreasing': {'color': '#f87171'}},
            number={'font': {'color': '#e2e8f0'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#374151", 'tickfont': {'color': '#6b7280'}},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': 'rgba(248,113,113,0.15)'},
                    {'range': [40, 70], 'color': 'rgba(251,191,36,0.15)'},
                    {'range': [70,100], 'color': 'rgba(52,211,153,0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#818cf8", 'width': 2},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Duration below gauge
        st.metric("Duration", f"{features_dict['duration']:.1f}s")

    # ── Key metrics ──
    st.markdown("### 📊 Key Speech Metrics")
    metrics = feedback_engine.get_metrics_summary(features_dict)
    cols = st.columns(3)
    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{metric_name}</strong>
                <span style="font-size:1.5rem;">{metric_value}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Feedback ──
    st.markdown("### 💡 AI-Powered Feedback")
    fb = feedback_engine.generate_feedback(features_dict, fluency_label, confidence)

    st.markdown(f"""
    <div class="feedback-box">
        <h4>📝 Summary</h4>
        <p>{fb['summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Strengths")
        if fb['strengths']:
            for s in fb['strengths']:
                st.markdown(f'<div class="success-box"><strong>✓</strong> {s}</div>', unsafe_allow_html=True)
        else:
            st.info("Keep practicing to build strengths!")
    with col2:
        st.markdown("#### 📈 Areas to Improve")
        if fb['improvements']:
            for i in fb['improvements']:
                st.markdown(f'<div class="warning-box"><strong>⚡</strong> {i}</div>', unsafe_allow_html=True)
        else:
            st.success("Excellent! No major areas for improvement.")

    st.markdown("#### 📋 Detailed Feedback")
    for detail in fb['detailed_feedback']:
        st.markdown(f"• {detail}")

    if fb['action_items']:
        st.markdown("#### 🎯 Action Items")
        for idx, action in enumerate(fb['action_items'], 1):
            st.markdown(f"{idx}. {action}")

    # ── Save history ──
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'fluency': fluency_label,
        'confidence': confidence,
        'duration': features_dict['duration'],
        'speech_rate': features_dict['speech_rate'],
        'energy': features_dict['energy_mean'],
        'score': score_value
    }
    if display_label:
        history_entry['source'] = display_label
    st.session_state.history.append(history_entry)
    pd.DataFrame(st.session_state.history).to_csv('user_history.csv', index=False)
    st.success("✅ Analysis complete! Results saved to history.")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">🎤 Personal Speech Coach</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Real-Time Fluency Analyzer</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:2.5px;color:#374151;margin-bottom:0.5rem;padding-left:0.2rem;font-weight:600;">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "page",
        ["🏠 Home", "🎙️ Analyze Speech", "🤖 Train Models", "📈 Analytics Dashboard", "📚 Help"],
        label_visibility="collapsed"
    )
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.25),transparent);margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;color:#374151;margin-bottom:0.4rem;padding-left:0.2rem;font-weight:600;">Supported Formats</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#4b5563;font-size:0.78rem;line-height:2;">' + " · ".join(f.upper() for f in SUPPORTED_FORMATS) + '</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("## Welcome to Your Personal Speech Coach! 🎯")

    # ── Row 1: Fluency Score Ranges + How It Works ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Fluency Score Ranges")
        fig = go.Figure()
        categories = ["Needs Improvement", "Average", "Fluent"]
        score_ranges = [40, 30, 30]   # widths: 0–40, 40–70, 70–100
        colors = ["#f87171", "#fbbf24", "#34d399"]
        for i, (cat, val, col) in enumerate(zip(categories, score_ranges, colors)):
            fig.add_trace(go.Bar(
                name=cat, x=[val], y=["Score Range"],
                orientation='h', marker_color=col,
                text=cat, textposition='inside',
                insidetextanchor='middle',
                hovertemplate=f"{cat}<extra></extra>"
            ))
        fig.update_layout(
            barmode='stack', height=130,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(l=0, r=0, t=10, b=30),
            xaxis=dict(
                tickvals=[0, 40, 70, 100],
                ticktext=["0", "40", "70", "100"],
                tickfont=dict(color='#6b7280'), gridcolor='rgba(55,65,81,0.4)'
            ),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        - 🟢 **Fluent (70–100)** – Excellent clarity and flow
        - 🟡 **Average (40–70)** – Good with room for improvement
        - 🔴 **Needs Improvement (0–40)** – Focus on key areas
        """)

    with col2:
        st.markdown("### 📊 How It Works")
        steps = ["Record / Upload", "Feature Extraction", "ML Prediction", "Feedback Generation", "Track Progress"]
        step_vals = [1, 1, 1, 1, 1]
        step_colors = ["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6"]
        fig2 = go.Figure(go.Bar(
            x=steps, y=step_vals,
            marker_color=step_colors,
            text=["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
            textposition='inside',
            insidetextanchor='middle',
            hovertemplate='<b>%{x}</b><extra></extra>'
        ))
        fig2.update_layout(
            height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(tickfont=dict(color='#6b7280', size=10), gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        1. **Record or Upload** your speech
        2. **Feature Extraction** – 40+ audio characteristics extracted
        3. **ML Prediction** – fluency level classified
        4. **Feedback Generation** – personalised tips generated
        5. **Track Progress** – monitor improvement over time
        """)

    st.markdown("---")

    # ── Row 2: Features radar + ML models bar ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🚀 Key Features Coverage")
        feature_labels = ["Live Recording", "Multi-Format", "MFCC Analysis", "ML Models", "AI Feedback", "Analytics", "CSV Export"]
        feature_vals   = [95, 90, 85, 88, 92, 80, 75]
        fig3 = go.Figure(go.Scatterpolar(
            r=feature_vals + [feature_vals[0]],
            theta=feature_labels + [feature_labels[0]],
            fill='toself',
            fillcolor='rgba(129,140,248,0.15)',
            line=dict(color='#818cf8', width=2),
            marker=dict(color='#c084fc', size=6)
        ))
        fig3.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#4b5563'), gridcolor='rgba(55,65,81,0.4)'),
                angularaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(55,65,81,0.3)')
            ),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
            height=280, margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("### 🤖 ML Models at a Glance")
        ml_models  = ["Logistic Regression", "Random Forest", "SVM"]
        ml_speed   = [95, 60, 70]   # relative speed
        ml_accuracy= [75, 90, 88]   # relative accuracy
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Speed', x=ml_models, y=ml_speed,   marker_color='#818cf8'))
        fig4.add_trace(go.Bar(name='Accuracy', x=ml_models, y=ml_accuracy, marker_color='#34d399'))
        fig4.update_layout(
            barmode='group', height=280,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=20, b=10),
            xaxis=dict(tickfont=dict(color='#6b7280'), gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(tickfont=dict(color='#6b7280'), gridcolor='rgba(55,65,81,0.4)', range=[0,110]),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.info("👈 Use the sidebar to navigate to **Analyze Speech** to get started, or **Train Models** to build the ML models first!")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE SPEECH PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎙️ Analyze Speech":
    st.markdown("## 🎙️ Speech Analysis")

    # ── Model check ──
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not yet trained! Please go to 'Train Models' first or load existing models.")
        if st.button("Load Pre-trained Models"):
            trainer = SpeechModelTrainer()
            if trainer.load_models('models'):
                st.session_state.trainer = trainer
                st.session_state.models_trained = True
                st.success("✅ Models loaded successfully!")
                st.rerun()
            else:
                st.error("❌ No pre-trained models found. Please train models first.")

    else:
        tab_live, tab_upload = st.tabs(["🎙️ Live Recording", "📁 Upload Audio File"])

        with tab_live:
            st.markdown("""
            <div class="recorder-box">
                <h3>🎙️ Record Directly in Your Browser</h3>
                <p>Click the microphone button below, speak clearly for <strong>15–30 seconds</strong>, then click stop. Your recording will be analysed instantly.</p>
            </div>
            """, unsafe_allow_html=True)

            col_tip1, col_tip2, col_tip3 = st.columns(3)
            with col_tip1:
                st.info("🔇 **Quiet room** reduces noise")
            with col_tip2:
                st.info("🎙️ **15–30 seconds** is ideal")
            with col_tip3:
                st.info("📢 **Speak naturally** at a comfortable pace")

            try:
                recorded_audio = st.audio_input("🔴 Click to record your speech")
            except AttributeError:
                recorded_audio = None
                st.warning("⚠️ Live recording requires Streamlit ≥ 1.31. Please upgrade: `pip install streamlit --upgrade`")

            if recorded_audio is not None:
                st.audio(recorded_audio, format="audio/wav")
                st.success("✅ Recording captured! Click **Analyse Recording** to get your results.")

                if st.button("🔍 Analyse Recording", type="primary", key="analyse_live"):
                    raw_bytes = recorded_audio.getvalue()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    run_analysis(tmp_path, display_label="Live Recording")
                    os.unlink(tmp_path)

        with tab_upload:
            st.markdown(f"""
            Upload an audio file in **any of these formats**:  
            {' · '.join(f'`{f.upper()}`' for f in SUPPORTED_FORMATS)}
            """)

            uploaded_file = st.file_uploader(
                "Choose your audio file",
                type=SUPPORTED_FORMATS,
                help="Drag and drop or click to browse. WAV, MP3, OGG, FLAC, M4A, AAC, WMA, AIFF, WEBM all supported."
            )

            if uploaded_file is not None:
                st.audio(uploaded_file)
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

                if file_ext not in ['.wav']:
                    st.info(f"ℹ️ Converting {file_ext.upper()} → WAV for analysis…")

                if st.button("🔍 Analyse Speech", type="primary", key="analyse_upload"):
                    file_bytes = uploaded_file.getvalue()

                    if file_ext == '.wav':
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(file_bytes)
                            wav_path = tmp.name
                    else:
                        wav_path = convert_to_wav(file_bytes, uploaded_file.name)

                    if wav_path:
                        run_analysis(wav_path, display_label=uploaded_file.name)
                        if wav_path != uploaded_file.name:
                            try:
                                os.unlink(wav_path)
                            except Exception:
                                pass

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("## 🤖 Model Training & Comparison")

    st.info("""
    This section trains and compares three machine learning models:
    - **Logistic Regression** – Simple, fast, interpretable
    - **Random Forest** – Ensemble method, handles non-linearity
    - **Support Vector Machine (SVM)** – Powerful for complex classification
    """)

    # ── Model overview charts ──
    _c1, _c2 = st.columns(2)
    with _c1:
        _fig_m = go.Figure()
        _attrs = ["Speed", "Accuracy", "Interpretability", "Non-linearity", "Robustness"]
        _models_info = {
            "Logistic Reg.": [95, 72, 98, 30, 70],
            "Random Forest": [60, 92, 55, 90, 88],
            "SVM":           [65, 88, 50, 80, 85],
        }
        _cols_m = ["#818cf8", "#34d399", "#f472b6"]
        for (mname, mvals), mcol in zip(_models_info.items(), _cols_m):
            _fig_m.add_trace(go.Scatterpolar(
                r=mvals + [mvals[0]], theta=_attrs + [_attrs[0]],
                fill='toself', name=mname,
                line=dict(color=mcol, width=2),
                fillcolor=f"rgba(129,140,248,0.08)" if mcol=="#818cf8" else (f"rgba(52,211,153,0.08)" if mcol=="#34d399" else f"rgba(244,114,182,0.08)")
            ))
        _fig_m.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color='#4b5563', size=9), gridcolor='rgba(55,65,81,0.4)'),
                angularaxis=dict(tickfont=dict(color='#94a3b8', size=10), gridcolor='rgba(55,65,81,0.3)')
            ),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            height=260, margin=dict(l=30, r=30, t=30, b=10),
            title=dict(text="Model Strengths Comparison", font=dict(color='#e2e8f0', size=13))
        )
        st.plotly_chart(_fig_m, use_container_width=True)
    with _c2:
        _fig_ds = go.Figure(data=[go.Pie(
            labels=["Training (80%)", "Testing (20%)"],
            values=[80, 20], hole=0.55,
            marker_colors=["#818cf8", "#f472b6"],
            textfont=dict(color='#e2e8f0')
        )])
        _fig_ds.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=260,
            font=dict(color='#94a3b8'),
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=20, r=20, t=30, b=10),
            title=dict(text="Dataset Split", font=dict(color='#e2e8f0', size=13))
        )
        st.plotly_chart(_fig_ds, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Number of training samples:", 100, 500, 300, 50)
    with col2:
        st.markdown("### 📊 Dataset Split")
        st.markdown("- Training: 80%")
        st.markdown("- Testing: 20%")

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("🔄 Training models… This may take a minute."):
            trainer = SpeechModelTrainer()
            progress_bar = st.progress(0)

            st.write("Generating synthetic dataset…")
            progress_bar.progress(20)
            df = trainer.create_synthetic_dataset(n_samples=n_samples)

            st.write(f"✅ Dataset created: {df.shape[0]} samples")
            st.dataframe(df['fluency_label'].value_counts())
            progress_bar.progress(40)

            st.write("Training models…")
            results = trainer.train_models(df)
            progress_bar.progress(80)

            trainer.save_models('models')
            st.session_state.trainer = trainer
            st.session_state.models_trained = True
            progress_bar.progress(100)

            st.success("✅ All models trained successfully!")

            # Comparison table
            st.markdown("---")
            st.markdown("## 📊 Model Comparison Results")
            comparison_data = [
                {
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1_score']:.4f}",
                }
                for model_name, metrics in results.items()
            ]
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

            # Bar chart — 2D only
            st.markdown("### 📈 Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#07090f')
            ax.set_facecolor('#07090f')
            models_list = list(results.keys())
            accuracies = [results[m]['accuracy'] * 100 for m in models_list]
            bar_colors = ['#818cf8', '#a855f7', '#c084fc']
            bars = ax.bar(models_list, accuracies, color=bar_colors, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.2f}%',
                        ha='center', va='bottom', fontweight='bold', color='#e2e8f0')
            ax.set_ylabel('Accuracy (%)', fontsize=12, color='#6b7280')
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', color='#e2e8f0')
            ax.set_ylim([0, 110])
            ax.grid(axis='y', alpha=0.15, color='#374151')
            ax.tick_params(colors='#4b5563')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1f2937')
            st.pyplot(fig)
            plt.close()

            st.markdown("### 🏆 Best Model Selected")
            st.success(f"**{trainer.best_model_name}** – Accuracy: {results[trainer.best_model_name]['accuracy']:.4f}")

            # Confusion matrices
            st.markdown("### 🔍 Confusion Matrices")
            cols = st.columns(len(results))
            for idx, (model_name, metrics) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"**{model_name}**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_facecolor('#07090f')
                    ax.set_facecolor('#07090f')
                    cm = metrics['confusion_matrix']
                    im = ax.imshow(cm, cmap='Purples')
                    ax.set_xticks(np.arange(len(trainer.label_encoder.classes_)))
                    ax.set_yticks(np.arange(len(trainer.label_encoder.classes_)))
                    ax.set_xticklabels(trainer.label_encoder.classes_, rotation=45, color='#6b7280')
                    ax.set_yticklabels(trainer.label_encoder.classes_, color='#6b7280')
                    for i in range(len(cm)):
                        for j in range(len(cm)):
                            ax.text(j, i, cm[i, j], ha="center", va="center", color="#e2e8f0")
                    ax.set_title("Confusion Matrix", color='#e2e8f0')
                    ax.set_xlabel('Predicted', color='#6b7280')
                    ax.set_ylabel('Actual', color='#6b7280')
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    plt.close()

    if st.session_state.models_trained:
        st.markdown("---")
        if st.button("🔄 Retrain Models with Different Parameters"):
            st.session_state.models_trained = False
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics Dashboard":
    st.markdown("## 📈 Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.info("📊 No analysis history yet. Analyse some speech samples to see your progress!")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        st.markdown("### 📊 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", len(history_df))
        with col2:
            st.metric("Fluent Speeches", len(history_df[history_df['fluency'] == 'Fluent']))
        with col3:
            st.metric("Avg Confidence", f"{history_df['confidence'].mean():.1f}%")
        with col4:
            st.metric("Avg Score", f"{history_df['score'].mean():.1f}")

        st.markdown("### 📈 Progress Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df.index, y=history_df['score'],
            mode='lines+markers', name='Fluency Score',
            line=dict(color='#818cf8', width=3), marker=dict(size=10, color='#c084fc')
        ))
        fig.update_layout(
            title="Fluency Score Progress", xaxis_title="Analysis Number",
            yaxis_title="Score", height=400, hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='rgba(55,65,81,0.5)'),
            yaxis=dict(gridcolor='rgba(55,65,81,0.5)')
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Fluency Distribution")
            fluency_counts = history_df['fluency'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=fluency_counts.index, values=fluency_counts.values, hole=.3,
                marker_colors=['#34d399', '#fbbf24', '#f87171']
            )])
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### ⚡ Speech Rate Trends")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=history_df.index, y=history_df['speech_rate'], marker_color='#a855f7'))
            fig.update_layout(
                xaxis_title="Analysis Number", yaxis_title="Speech Rate (BPM)", height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis=dict(gridcolor='rgba(55,65,81,0.5)'),
                yaxis=dict(gridcolor='rgba(55,65,81,0.5)')
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Analysis History")
        display_cols = ['timestamp', 'fluency', 'confidence', 'duration', 'speech_rate', 'score']
        if 'source' in history_df.columns:
            display_cols.insert(1, 'source')
        st.dataframe(history_df[display_cols], use_container_width=True)

        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History (CSV)", data=csv,
            file_name='speech_analysis_history.csv', mime='text/csv'
        )

        if st.button("🗑️ Clear History", type="secondary"):
            if st.checkbox("Are you sure?"):
                st.session_state.history = []
                if os.path.exists('user_history.csv'):
                    os.remove('user_history.csv')
                st.success("History cleared!")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# HELP PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Help":
    st.markdown("## 📚 Help & Documentation")

    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "Features", "Tips", "Troubleshooting"])

    with tab1:
        st.markdown("### 🚀 Getting Started")
        # Steps funnel chart
        _steps = ["1. Train Models", "2. Analyse Speech", "3. Track Progress"]
        _step_vals = [100, 80, 60]
        _step_colors = ["#818cf8", "#a855f7", "#f472b6"]
        _fig_steps = go.Figure(go.Bar(
            x=_steps, y=_step_vals, marker_color=_step_colors,
            text=["Train", "Analyse", "Track"], textposition='inside',
            insidetextanchor='middle', textfont=dict(color='#fff', size=13),
            hovertemplate='<b>%{x}</b><extra></extra>'
        ))
        _fig_steps.update_layout(
            height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(_fig_steps, use_container_width=True)
        st.markdown("""
        #### Step 1: Train Models
        1. Navigate to **Train Models** page
        2. Choose number of training samples (300 recommended)
        3. Click **Train Models** – takes about 1–2 minutes

        #### Step 2: Analyse Speech
        **Option A – Live Recording:**
        1. Go to **Analyse Speech → 🎙️ Live Recording**
        2. Click the microphone button and speak for 15–30 seconds
        3. Click **Analyse Recording**

        **Option B – File Upload:**
        1. Go to **Analyse Speech → 📁 Upload Audio File**
        2. Upload any audio file (WAV, MP3, OGG, FLAC, M4A…)
        3. Click **Analyse Speech**

        #### Step 3: Track Progress
        1. Visit **Analytics Dashboard**
        2. View your progress over time
        3. Download history as CSV
        """)

    with tab2:
        st.markdown("### 🎯 Features Explained")
        st.markdown("#### Supported Audio Formats")
        st.markdown(f"{' · '.join(f'`{f.upper()}`' for f in SUPPORTED_FORMATS)}")
        # Format chart
        _fmt_fig = go.Figure(data=[go.Pie(
            labels=[f.upper() for f in SUPPORTED_FORMATS],
            values=[1]*len(SUPPORTED_FORMATS), hole=0.4,
            marker_colors=["#818cf8","#a855f7","#c084fc","#f472b6","#34d399","#fbbf24","#f87171","#60a5fa","#a3e635"],
            textfont=dict(color='#fff', size=11)
        )])
        _fmt_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=200,
            font=dict(color='#94a3b8'), showlegend=True,
            legend=dict(font=dict(color='#94a3b8', size=10), bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.15),
            margin=dict(l=0, r=0, t=10, b=40),
            title=dict(text="Supported Formats", font=dict(color='#e2e8f0', size=12))
        )
        st.plotly_chart(_fmt_fig, use_container_width=True)
        st.markdown("""
        **Automatic conversion** – non-WAV files are silently converted using pydub + ffmpeg before analysis.

        #### Audio Features Extracted
        """)
        # Audio features importance bar
        _feat_names = ["MFCC (x13)", "Pitch", "Energy", "Speech Rate", "Silence Ratio", "ZCR", "Spectral"]
        _feat_vals  = [13, 4, 2, 1, 1, 1, 3]
        _feat_fig = go.Figure(go.Bar(
            x=_feat_vals, y=_feat_names, orientation='h',
            marker_color=["#818cf8","#a855f7","#c084fc","#f472b6","#34d399","#fbbf24","#f87171"],
            text=[f"{v} feature{'s' if v>1 else ''}" for v in _feat_vals],
            textposition='outside', textfont=dict(color='#94a3b8', size=11),
            hovertemplate='<b>%{y}</b>: %{x} features<extra></extra>'
        ))
        _feat_fig.update_layout(
            height=240, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(visible=False), yaxis=dict(tickfont=dict(color='#94a3b8'))
        )
        st.plotly_chart(_feat_fig, use_container_width=True)
        st.markdown("""
        - **MFCC**: Mel-frequency cepstral coefficients (voice timbre & characteristics)
        - **Pitch**: Fundamental frequency – mean, std, max, min
        - **Energy**: Volume and vocal projection (RMS)
        - **Speech Rate**: Tempo / pace in BPM
        - **Pause Duration / Silence Ratio**: How much you pause
        - **Zero Crossing Rate**: Voice quality indicator
        - **Spectral Features**: Centroid, rolloff, bandwidth

        #### ML Models
        - **Logistic Regression** – fast linear classifier
        - **Random Forest** – ensemble of decision trees
        - **SVM** – support vector machine for complex decision boundaries
        """)

    with tab3:
        st.markdown("### 💡 Tips for Better Results")
        # Ideal speech parameters chart
        _tip_params = ["Speech Rate (BPM)", "Duration (sec)", "Distance (cm)", "Noise Level (%)"]
        _tip_ideal  = [110, 22, 22, 5]
        _tip_min    = [100, 15, 15, 0]
        _tip_max    = [120, 30, 30, 15]
        _tip_fig = go.Figure()
        _tip_fig.add_trace(go.Bar(name='Min', x=_tip_params, y=_tip_min, marker_color='rgba(129,140,248,0.3)'))
        _tip_fig.add_trace(go.Bar(name='Ideal', x=_tip_params, y=_tip_ideal, marker_color='#34d399'))
        _tip_fig.add_trace(go.Bar(name='Max', x=_tip_params, y=_tip_max, marker_color='rgba(244,114,182,0.3)'))
        _tip_fig.update_layout(
            barmode='overlay', height=220,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=20, b=10),
            xaxis=dict(tickfont=dict(color='#94a3b8', size=10), gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(tickfont=dict(color='#6b7280'), gridcolor='rgba(55,65,81,0.3)'),
            title=dict(text="Ideal Recording Parameters", font=dict(color='#e2e8f0', size=13))
        )
        st.plotly_chart(_tip_fig, use_container_width=True)
        st.markdown("""
        #### Recording Tips
        - Use a good quality microphone or headset
        - Record in a quiet room (close windows, turn off fans)
        - Speak for 15–30 seconds for the best analysis
        - Stay 15–30 cm from the microphone

        #### Speaking Tips
        - Moderate pace: 100–120 BPM
        - Use natural pauses between ideas
        - Project your voice with good energy
        - Vary your pitch for expressiveness

        #### File Tips
        - Any common format is accepted (WAV is natively supported; others auto-converted)
        - Duration: 5–60 seconds (15–30 s ideal)
        - Clear audio with minimal background noise gives the best results
        """)

    with tab4:
        st.markdown("### 🔧 Troubleshooting")
        # Common issues chart
        _issues = ["Live Recording", "File Conversion", "Models Not Trained", "Poor Results"]
        _issue_freq = [40, 25, 20, 15]
        _issue_colors = ["#f87171", "#fbbf24", "#818cf8", "#34d399"]
        _issue_fig = go.Figure(go.Bar(
            x=_issue_freq, y=_issues, orientation='h',
            marker_color=_issue_colors,
            text=[f"{v}% of issues" for v in _issue_freq],
            textposition='outside', textfont=dict(color='#94a3b8', size=11),
            hovertemplate='<b>%{y}</b>: %{x}%<extra></extra>'
        ))
        _issue_fig.update_layout(
            height=190, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(l=0, r=100, t=20, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(color='#94a3b8')),
            title=dict(text="Common Issues Breakdown", font=dict(color='#e2e8f0', size=13))
        )
        st.plotly_chart(_issue_fig, use_container_width=True)

        with st.expander("Live recording not working"):
            st.markdown("""
            **Solution:**
            1. Upgrade Streamlit: `pip install streamlit --upgrade` (requires ≥ 1.31)
            2. Allow microphone access in your browser when prompted
            3. Use a modern browser (Chrome / Edge / Firefox)
            """)

        with st.expander("Non-WAV file conversion fails"):
            st.markdown("""
            **Solution:**
            - Install ffmpeg: `sudo apt install ffmpeg` (Linux) or download from https://ffmpeg.org
            - On Windows: add ffmpeg to your PATH
            - Alternatively, convert your file to WAV manually using Audacity
            """)

        with st.expander("Models not trained"):
            st.markdown("""
            **Solution:**
            1. Go to **Train Models** page → click **Train Models**
            2. Or click **Load Pre-trained Models** on the Analyse page
            """)

        with st.expander("Poor / unexpected results"):
            st.markdown("""
            **Solution:**
            1. Check recording quality
            2. Reduce background noise
            3. Speak louder and more clearly
            4. Ensure correct duration (15–30 s)
            5. Try one of the sample files to verify the app is working
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#374151;padding:1.5rem 0;font-size:0.85rem;">
    <p>🎤 Personal Speech Coach – AI Powered Real-Time Fluency Analyzer</p>
    <p>Built with ❤️ using Streamlit · Librosa · Scikit-learn · pydub</p>
</div>
""", unsafe_allow_html=True)