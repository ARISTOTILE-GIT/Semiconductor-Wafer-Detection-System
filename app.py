import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch
from PIL import Image as PILImage
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import torchvision.transforms as transforms
import streamlit.components.v1 as components

from model_utils import load_model, predict, murphy_yield, CLASSES
from nlp_utils import get_explanation

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern React Native–inspired CSS ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F0F2F8;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── App header banner ── */
    .app-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 60%, #6D28D9 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 8px 32px rgba(55,48,163,0.25);
    }
    .app-header h1 {
        color: #FFFFFF;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.65);
        font-size: 0.85rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Cards ── */
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1E1B4B;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #E0E7FF !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(239,68,68,0.18) !important;
        border: 1px solid rgba(239,68,68,0.4) !important;
        color: #FCA5A5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(239,68,68,0.35) !important;
    }

    /* ── Sidebar stat pills ── */
    .stat-pill {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stat-pill .label { font-size: 0.78rem; color: rgba(224,231,255,0.7); }
    .stat-pill .value { font-size: 0.95rem; font-weight: 700; color: #E0E7FF; }

    /* ── Decision badges ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 1.1rem;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-save   { background: #D1FAE5; color: #065F46; }
    .badge-review { background: #FEF3C7; color: #92400E; }
    .badge-scrap  { background: #FEE2E2; color: #991B1B; }

    /* ── Metric cards ── */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-card .m-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.3rem;
    }
    .metric-card .m-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1E293B;
    }

    /* ── Expert analysis box ── */
    .expert-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
        border-left: 4px solid #6D28D9;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #3730A3;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 0.3rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        gap: 0.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #6B7280;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #3730A3 !important;
        color: #FFFFFF !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #FAFAFA;
        border: 2px dashed #C7D2FE;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3730A3, #6D28D9) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 4px 14px rgba(109,40,217,0.35) !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stAlert { border-radius: 10px !important; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Groq key from Streamlit secrets ───────────────────────────
groq_key = st.secrets.get("GROQ_API_KEY", "")

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def get_model():
    with st.spinner("Loading model from HuggingFace..."):
        return load_model()

model = get_model()

# ── Session State ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size:2.2rem;">🔬</div>
            <div style="font-size:1rem; font-weight:700; color:#E0E7FF; margin-top:0.3rem;">Wafer Defect System</div>
            <div style="font-size:0.72rem; color:rgba(224,231,255,0.5); margin-top:0.2rem;">EfficientNet-B3 · Murphy Yield</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:600;letter-spacing:0.8px;color:rgba(224,231,255,0.5);text-transform:uppercase;margin-bottom:0.5rem;">Wafer Config</p>', unsafe_allow_html=True)
    wafer_id = st.text_input("Wafer ID", value="WAFER-001")
    die_area = st.slider("Die Area (cm²)", 0.1, 2.0, 0.5, 0.1)
    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:600;letter-spacing:0.8px;color:rgba(224,231,255,0.5);text-transform:uppercase;margin-bottom:0.5rem;">Session Stats</p>', unsafe_allow_html=True)
    total = len(st.session_state.history)
    if total > 0:
        saves     = sum(1 for h in st.session_state.history if h["Decision"] == "SAVE")
        reviews   = sum(1 for h in st.session_state.history if h["Decision"] == "REVIEW")
        scraps    = sum(1 for h in st.session_state.history if h["Decision"] == "SCRAP")
        avg_yield = round(sum(h["Yield (%)"] for h in st.session_state.history) / total, 2)
        st.markdown(f"""
        <div class="stat-pill"><span class="label">Total Wafers</span><span class="value">{total}</span></div>
        <div class="stat-pill"><span class="label">Avg Yield</span><span class="value">{avg_yield}%</span></div>
        <div class="stat-pill"><span class="label">✅ Save</span><span class="value" style="color:#6EE7B7">{saves}</span></div>
        <div class="stat-pill"><span class="label">⚠️ Review</span><span class="value" style="color:#FCD34D">{reviews}</span></div>
        <div class="stat-pill"><span class="label">❌ Scrap</span><span class="value" style="color:#FCA5A5">{scraps}</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.82rem;color:rgba(224,231,255,0.4);text-align:center;padding:0.5rem 0;">No predictions yet</p>', unsafe_allow_html=True)

    st.write("")
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── App Header ─────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Semiconductor Wafer Defect Detection</h1>
    <p>EfficientNet-B3 &nbsp;·&nbsp; Murphy Yield Model &nbsp;·&nbsp; LLM Expert Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["  Prediction  ", "  Yield Analysis  ", "  History  ", "  Draw Mode  "])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.markdown('<div class="card"><div class="card-title">Upload Wafer Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose wafer image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded:
            image = PILImage.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Wafer", use_container_width=True)
            predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")
        else:
            st.markdown('<p style="color:#94A3B8;font-size:0.85rem;text-align:center;padding:1rem 0;">Drop a wafer image to begin</p>', unsafe_allow_html=True)
            predict_btn = False
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if uploaded and predict_btn:
            with st.spinner("Analyzing wafer..."):
                result = predict(image, model, die_area)

            defect = result["defect_type"]
            conf   = result["confidence"]
            yld    = result["yield_pct"]
            dec    = result["decision"]
            probs  = result["all_probs"]

            badge_map = {
                "SAVE":   ("badge-save",   "✅ SAVE"),
                "REVIEW": ("badge-review", "⚠️ REVIEW"),
                "SCRAP":  ("badge-scrap",  "❌ SCRAP"),
            }
            bcls, blabel = badge_map[dec]
            st.markdown(f'<span class="badge {bcls}">{blabel}</span>', unsafe_allow_html=True)
            st.write("")

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="m-label">Defect Type</div><div class="m-value">{defect}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="m-label">Confidence</div><div class="m-value">{conf:.1f}%</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="m-label">Murphy Yield</div><div class="m-value">{yld}%</div></div>', unsafe_allow_html=True)

            color_map = {"SAVE": "#10B981", "REVIEW": "#F59E0B", "SCRAP": "#EF4444"}
            bar_color = color_map[dec]
            fig, ax = plt.subplots(figsize=(6, 1.0))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8FAFC")
            ax.barh([""], [yld],       color=bar_color, height=0.55, edgecolor="none")
            ax.barh([""], [100 - yld], left=[yld], color="#E2E8F0", height=0.55, edgecolor="none")
            ax.axvline(x=70, color="#10B981", linestyle="--", linewidth=1.5, label="70% Save")
            ax.axvline(x=40, color="#F59E0B", linestyle="--", linewidth=1.5, label="40% Min")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Yield %", fontsize=8, color="#6B7280")
            ax.tick_params(colors="#6B7280", labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown('<p style="font-size:0.75rem;font-weight:600;color:#64748B;margin-top:0.8rem;margin-bottom:0.2rem;">CLASS PROBABILITIES</p>', unsafe_allow_html=True)
            prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability (%)"])
            prob_df = prob_df.sort_values("Probability (%)", ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor("#FFFFFF")
            ax2.set_facecolor("#F8FAFC")
            ax2.barh(prob_df["Class"], prob_df["Probability (%)"],
                     color=["#6D28D9" if c == defect else "#CBD5E1" for c in prob_df["Class"]],
                     edgecolor="none", height=0.6)
            ax2.set_xlabel("Probability (%)", fontsize=8, color="#6B7280")
            ax2.tick_params(colors="#374151", labelsize=8)
            ax2.set_xlim(0, 100)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            if groq_key:
                st.markdown('<div class="card-title" style="margin-top:0.8rem;">Expert Analysis</div>', unsafe_allow_html=True)
                with st.spinner("Fetching expert analysis..."):
                    explanation = get_explanation(defect, conf, yld, dec, groq_key)
                st.markdown(f'<div class="expert-box">{explanation.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Wafer ID":       wafer_id,
                "Defect":         defect,
                "Confidence (%)": round(conf, 2),
                "Yield (%)":      yld,
                "Decision":       dec
            })

        elif not uploaded:
            st.markdown('<div class="card"><p style="color:#94A3B8;text-align:center;padding:3rem 1rem;font-size:0.9rem;">Results will appear here after prediction</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — YIELD ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Last Prediction Context</div>
            <span style="font-size:0.9rem;color:#374151;">Defect: <strong>{last['Defect']}</strong>&nbsp;&nbsp;·&nbsp;&nbsp;Yield: <strong>{last['Yield (%)']}%</strong></span>
        </div>
        """, unsafe_allow_html=True)

        die_areas  = np.linspace(0.1, 2.0, 50)
        last_ratio = last["Yield (%)"] / 100
        yields = []
        for da in die_areas:
            import math
            num_dies = 706.86 / da
            dpd = last_ratio * num_dies * da / 706.86
            if dpd <= 0:
                yields.append(100.0)
            elif dpd > 10:
                yields.append(0.0)
            else:
                yields.append(round(((1 - math.exp(-dpd)) / dpd) ** 2 * 100, 2))

        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FAFBFF")
        ax.plot(die_areas, yields, color="#6D28D9", linewidth=2.5, zorder=3)
        ax.axhline(y=70, color="#10B981", linestyle="--", linewidth=1.5, label="70% Save threshold")
        ax.axhline(y=40, color="#F59E0B", linestyle="--", linewidth=1.5, label="40% Review threshold")
        ax.axvline(x=die_area, color="#EF4444", linestyle="-", linewidth=2, label=f"Current die area ({die_area} cm²)")
        ax.fill_between(die_areas, yields, 0, alpha=0.08, color="#6D28D9")
        ax.set_xlabel("Die Area (cm²)", color="#6B7280", fontsize=9)
        ax.set_ylabel("Yield (%)", color="#6B7280", fontsize=9)
        ax.set_title(f"Murphy Yield Curve — {last['Defect']} Defect", fontsize=11, fontweight="bold", color="#1E293B")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15, color="#94A3B8")
        ax.tick_params(colors="#6B7280", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(st.session_state.history) > 1:
            st.markdown('<div class="card-title" style="margin-top:1rem;">Average Yield by Defect Type</div>', unsafe_allow_html=True)
            df = pd.DataFrame(st.session_state.history)
            avg_by_defect = df.groupby("Defect")["Yield (%)"].mean().reset_index()
            fig3, ax3 = plt.subplots(figsize=(8, 3.5))
            fig3.patch.set_facecolor("#FFFFFF")
            ax3.set_facecolor("#FAFBFF")
            colors = ["#10B981" if y >= 70 else "#F59E0B" if y >= 40 else "#EF4444"
                      for y in avg_by_defect["Yield (%)"]]
            ax3.bar(avg_by_defect["Defect"], avg_by_defect["Yield (%)"], color=colors,
                    edgecolor="none", width=0.55)
            ax3.axhline(y=70, color="#10B981", linestyle="--", linewidth=1, label="70% Save")
            ax3.axhline(y=40, color="#F59E0B", linestyle="--", linewidth=1, label="40% Min")
            ax3.set_ylabel("Average Yield (%)", color="#6B7280", fontsize=9)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.15, axis="y", color="#94A3B8")
            ax3.tick_params(colors="#6B7280", labelsize=8)
            plt.xticks(rotation=30)
            for spine in ax3.spines.values():
                spine.set_edgecolor("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
    else:
        st.markdown('<div class="card"><p style="text-align:center;color:#94A3B8;padding:3rem;">Run at least one prediction to see yield analysis</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        def color_decision(val):
            return {
                "SAVE":   "background-color:#D1FAE5;color:#065F46;font-weight:600",
                "REVIEW": "background-color:#FEF3C7;color:#92400E;font-weight:600",
                "SCRAP":  "background-color:#FEE2E2;color:#991B1B;font-weight:600",
            }.get(val, "")

        st.dataframe(
            df.style.map(color_decision, subset=["Decision"]),
            use_container_width=True,
            height=300
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "wafer_history.csv",
            "text/csv"
        )

        st.write("")
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="card-title">Defect Distribution</div>', unsafe_allow_html=True)
            defect_counts = df["Defect"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            fig4.patch.set_facecolor("#FFFFFF")
            palette = ["#6D28D9","#3730A3","#7C3AED","#A78BFA","#C4B5FD","#DDD6FE","#EDE9FE","#F5F3FF"]
            ax4.pie(defect_counts.values, labels=defect_counts.index, autopct="%1.1f%%",
                    colors=palette[:len(defect_counts)], startangle=90,
                    wedgeprops={"edgecolor":"white","linewidth":2})
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        with col_b:
            st.markdown('<div class="card-title">Decision Breakdown</div>', unsafe_allow_html=True)
            dec_counts = df["Decision"].value_counts()
            dec_colors = {"SAVE": "#10B981", "REVIEW": "#F59E0B", "SCRAP": "#EF4444"}
            fig5, ax5 = plt.subplots(figsize=(5, 4))
            fig5.patch.set_facecolor("#FFFFFF")
            ax5.set_facecolor("#FAFBFF")
            ax5.bar(dec_counts.index, dec_counts.values,
                    color=[dec_colors.get(d, "#94A3B8") for d in dec_counts.index],
                    edgecolor="none", width=0.45)
            ax5.set_ylabel("Count", color="#6B7280", fontsize=9)
            ax5.tick_params(colors="#6B7280", labelsize=9)
            ax5.grid(True, alpha=0.15, axis="y")
            for spine in ax5.spines.values():
                spine.set_edgecolor("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

        st.markdown('<div class="card-title" style="margin-top:0.5rem;">Yield Trend</div>', unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(10, 2.8))
        fig6.patch.set_facecolor("#FFFFFF")
        ax6.set_facecolor("#FAFBFF")
        ax6.plot(range(len(df)), df["Yield (%)"], marker="o", markersize=5,
                 color="#6D28D9", linewidth=2.5, zorder=3)
        ax6.axhline(y=70, color="#10B981", linestyle="--", linewidth=1, label="70% Save")
        ax6.axhline(y=40, color="#F59E0B", linestyle="--", linewidth=1, label="40% Min")
        ax6.fill_between(range(len(df)), df["Yield (%)"], alpha=0.08, color="#6D28D9")
        ax6.set_xlabel("Prediction #", color="#6B7280", fontsize=9)
        ax6.set_ylabel("Yield (%)", color="#6B7280", fontsize=9)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.15, color="#94A3B8")
        ax6.tick_params(colors="#6B7280", labelsize=8)
        for spine in ax6.spines.values():
            spine.set_edgecolor("#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    else:
        st.markdown('<div class="card"><p style="text-align:center;color:#94A3B8;padding:3rem;">No history yet — run some predictions first</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — DRAW MODE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p style="color:#6B7280;font-size:0.88rem;margin-bottom:1rem;">Click or drag on the canvas to mark defect dies, then run prediction.</p>', unsafe_allow_html=True)

    col_draw, col_draw_result = st.columns([1, 1.2], gap="large")

    with col_draw:
        st.markdown('<div class="card-title">Defect Pattern Canvas</div>', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#C8C8C8",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
        cb1, cb2 = st.columns(2)
        with cb1:
            draw_predict = st.button("Run Prediction", use_container_width=True, type="primary")
        with cb2:
            st.button("Clear Canvas", use_container_width=True)

    with col_draw_result:
        if canvas_result.image_data is not None and draw_predict:
            canvas_array = canvas_result.image_data.astype(np.uint8)
            canvas_img   = PILImage.fromarray(canvas_array).convert("RGB")

            canvas_np   = np.array(canvas_img)
            h, w        = canvas_np.shape[:2]
            mask_circle = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_circle, (w // 2, h // 2), min(w, h) // 2 - 5, 255, -1)
            result_img  = np.where(mask_circle[:, :, None] > 0, canvas_np, 0)
            canvas_img  = PILImage.fromarray(result_img.astype(np.uint8))

            with st.spinner("Predicting..."):
                result = predict(canvas_img, model, die_area)

            defect = result["defect_type"]
            conf   = result["confidence"]
            yld    = result["yield_pct"]
            dec    = result["decision"]

            badge_map = {
                "SAVE":   ("badge-save",   "✅ SAVE"),
                "REVIEW": ("badge-review", "⚠️ REVIEW"),
                "SCRAP":  ("badge-scrap",  "❌ SCRAP"),
            }
            bcls, blabel = badge_map[dec]
            st.markdown(f'<span class="badge {bcls}">{blabel}</span>', unsafe_allow_html=True)
            st.write("")

            d1, d2, d3 = st.columns(3)
            d1.markdown(f'<div class="metric-card"><div class="m-label">Defect</div><div class="m-value" style="font-size:1rem;">{defect}</div></div>', unsafe_allow_html=True)
            d2.markdown(f'<div class="metric-card"><div class="m-label">Confidence</div><div class="m-value">{conf:.1f}%</div></div>', unsafe_allow_html=True)
            d3.markdown(f'<div class="metric-card"><div class="m-label">Yield</div><div class="m-value">{yld}%</div></div>', unsafe_allow_html=True)

            probs   = result["all_probs"]
            prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Prob"])
            prob_df = prob_df.sort_values("Prob", ascending=True)
            fig7, ax7 = plt.subplots(figsize=(5, 3.5))
            fig7.patch.set_facecolor("#FFFFFF")
            ax7.set_facecolor("#FAFBFF")
            ax7.barh(prob_df["Class"], prob_df["Prob"],
                     color=["#6D28D9" if c == defect else "#CBD5E1" for c in prob_df["Class"]],
                     edgecolor="none", height=0.6)
            ax7.set_xlabel("Probability (%)", fontsize=8, color="#6B7280")
            ax7.tick_params(colors="#374151", labelsize=8)
            for spine in ax7.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()

            if groq_key:
                with st.spinner("Fetching expert analysis..."):
                    explanation = get_explanation(defect, conf, yld, dec, groq_key)
                st.markdown(f'<div class="expert-box">{explanation.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Wafer ID":       wafer_id + "-DRAW",
                "Defect":         defect,
                "Confidence (%)": round(conf, 2),
                "Yield (%)":      yld,
                "Decision":       dec
            })
        else:
            st.markdown('<p style="color:#94A3B8;font-size:0.85rem;text-align:center;padding:2rem 0;">Draw a pattern and click Run Prediction</p>', unsafe_allow_html=True)
