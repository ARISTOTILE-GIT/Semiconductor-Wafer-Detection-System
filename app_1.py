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

from model_utils import load_model, predict, murphy_yield, CLASSES
from nlp_utils import get_explanation

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detection System",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .save-badge {
        background-color: #28a745;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .review-badge {
        background-color: #ffc107;
        color: black;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .scrap-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .expert-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.image("https://img.icons8.com/color/96/processor.png", width=80)
    st.title("⚙️ Settings")
    st.divider()

    groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    st.divider()

    wafer_id    = st.text_input("🏷️ Wafer ID", value="WAFER-001")
    die_area    = st.slider("📐 Die Area (cm²)", 0.1, 2.0, 0.5, 0.1)
    st.divider()

    st.markdown("**📊 Session Stats**")
    total = len(st.session_state.history)
    if total > 0:
        saves   = sum(1 for h in st.session_state.history if h["Decision"] == "SAVE")
        reviews = sum(1 for h in st.session_state.history if h["Decision"] == "REVIEW")
        scraps  = sum(1 for h in st.session_state.history if h["Decision"] == "SCRAP")
        avg_yield = round(sum(h["Yield (%)"] for h in st.session_state.history) / total, 2)
        st.metric("Total Wafers", total)
        st.metric("Avg Yield", f"{avg_yield}%")
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Save",   saves)
        col2.metric("⚠️ Review", reviews)
        col3.metric("❌ Scrap",  scraps)
    else:
        st.info("No predictions yet!")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ── Title ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Semiconductor Wafer Defect Detection System</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>EfficientNet-B3 | Murphy Yield Model | LLM Expert Analysis</p>", unsafe_allow_html=True)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📈 Yield Analysis", "📋 History", "🎨 Draw Mode"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.subheader("📤 Upload Wafer Image")
        uploaded = st.file_uploader("Choose wafer image", type=["png", "jpg", "jpeg"])

        if uploaded:
            image = PILImage.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Wafer", use_container_width=True)
            predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")
        else:
            st.info("Upload a wafer image to get started!")
            predict_btn = False

    with col_result:
        if uploaded and predict_btn:
            with st.spinner("Analyzing wafer..."):
                result = predict(image, model, die_area)

            defect  = result["defect_type"]
            conf    = result["confidence"]
            yld     = result["yield_pct"]
            dec     = result["decision"]
            probs   = result["all_probs"]

            # Decision badge
            st.subheader("📊 Results")
            badge_class = {"SAVE": "save-badge", "REVIEW": "review-badge", "SCRAP": "scrap-badge"}[dec]
            badge_emoji = {"SAVE": "✅", "REVIEW": "⚠️", "SCRAP": "❌"}[dec]
            st.markdown(f'<span class="{badge_class}">{badge_emoji} {dec}</span>', unsafe_allow_html=True)
            st.write("")

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("🦠 Defect Type", defect)
            m2.metric("🎯 Confidence",  f"{conf:.1f}%")
            m3.metric("📈 Murphy Yield", f"{yld}%")

            # Yield bar
            st.write("")
            color = {"SAVE": "#28a745", "REVIEW": "#ffc107", "SCRAP": "#dc3545"}[dec]
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh(["Yield"], [yld],       color=color,      height=0.5)
            ax.barh(["Yield"], [100 - yld], left=[yld], color="#e9ecef", height=0.5)
            ax.set_xlim(0, 100)
            ax.axvline(x=70, color="green",  linestyle="--", linewidth=1.5, label="70% Save")
            ax.axvline(x=40, color="orange", linestyle="--", linewidth=1.5, label="40% Min")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("Yield %")
            ax.set_title(f"Murphy Yield: {yld}%")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # All probabilities
            st.write("**All Probabilities:**")
            prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability (%)"])
            prob_df = prob_df.sort_values("Probability (%)", ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bars = ax2.barh(prob_df["Class"], prob_df["Probability (%)"],
                            color=["#4361ee" if c == defect else "#adb5bd" for c in prob_df["Class"]])
            ax2.set_xlabel("Probability (%)")
            ax2.set_title("Class Probabilities")
            ax2.set_xlim(0, 100)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # NLP Explanation
            if groq_key:
                st.write("")
                st.subheader("🧠 Expert Analysis")
                with st.spinner("Getting expert analysis..."):
                    explanation = get_explanation(defect, conf, yld, dec, groq_key)
                st.markdown(f'<div class="expert-box">{explanation.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
            else:
                st.warning("Add Groq API key in sidebar for expert analysis!")

            # Save to history
            st.session_state.history.append({
                "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Wafer ID":    wafer_id,
                "Defect":      defect,
                "Confidence (%)": round(conf, 2),
                "Yield (%)":   yld,
                "Decision":    dec
            })

# ══════════════════════════════════════════════════════════════
# TAB 2 — YIELD ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Yield vs Die Area")

    if st.session_state.history:
        last = st.session_state.history[-1]
        st.info(f"Showing yield analysis for last prediction: **{last['Defect']}** | Yield: **{last['Yield (%)']}%**")

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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(die_areas, yields, color="#4361ee", linewidth=2.5)
        ax.axhline(y=70, color="green",  linestyle="--", linewidth=1.5, label="70% SAVE threshold")
        ax.axhline(y=40, color="orange", linestyle="--", linewidth=1.5, label="40% REVIEW threshold")
        ax.axvline(x=die_area, color="red", linestyle="-", linewidth=2, label=f"Current die area ({die_area} cm²)")
        ax.fill_between(die_areas, yields, 0, alpha=0.1, color="#4361ee")
        ax.set_xlabel("Die Area (cm²)")
        ax.set_ylabel("Yield (%)")
        ax.set_title(f"Murphy Yield Curve — {last['Defect']} Defect")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Defect type yield comparison
        if len(st.session_state.history) > 1:
            st.subheader("📊 Yield by Defect Type")
            df = pd.DataFrame(st.session_state.history)
            avg_by_defect = df.groupby("Defect")["Yield (%)"].mean().reset_index()
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            colors = ["#28a745" if y >= 70 else "#ffc107" if y >= 40 else "#dc3545"
                      for y in avg_by_defect["Yield (%)"]]
            ax3.bar(avg_by_defect["Defect"], avg_by_defect["Yield (%)"], color=colors)
            ax3.axhline(y=70, color="green",  linestyle="--", linewidth=1, label="70% Save")
            ax3.axhline(y=40, color="orange", linestyle="--", linewidth=1, label="40% Min")
            ax3.set_ylabel("Average Yield (%)")
            ax3.set_title("Average Yield by Defect Type")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
    else:
        st.info("Run at least one prediction to see yield analysis!")

# ══════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        # Color decision column
        def color_decision(val):
            colors = {"SAVE": "background-color: #d4edda", 
                      "REVIEW": "background-color: #fff3cd",
                      "SCRAP": "background-color: #f8d7da"}
            return colors.get(val, "")

        st.dataframe(
            df.style.applymap(color_decision, subset=["Decision"]),
            use_container_width=True
        )

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "wafer_history.csv",
            "text/csv",
            use_container_width=True
        )

        st.divider()

        # Charts
        col_a, col_b = st.columns(2)

        with col_a:
            st.write("**Defect Distribution**")
            defect_counts = df["Defect"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            ax4.pie(defect_counts.values, labels=defect_counts.index, autopct="%1.1f%%",
                    colors=plt.cm.Set3.colors)
            ax4.set_title("Defect Type Distribution")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        with col_b:
            st.write("**Decision Distribution**")
            dec_counts = df["Decision"].value_counts()
            dec_colors = {"SAVE": "#28a745", "REVIEW": "#ffc107", "SCRAP": "#dc3545"}
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            ax5.bar(dec_counts.index,
                    dec_counts.values,
                    color=[dec_colors.get(d, "gray") for d in dec_counts.index])
            ax5.set_ylabel("Count")
            ax5.set_title("SAVE / REVIEW / SCRAP Count")
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

        # Yield trend
        st.write("**Yield Trend Over Time**")
        fig6, ax6 = plt.subplots(figsize=(10, 3))
        ax6.plot(range(len(df)), df["Yield (%)"], marker="o", color="#4361ee", linewidth=2)
        ax6.axhline(y=70, color="green",  linestyle="--", linewidth=1, label="70% Save")
        ax6.axhline(y=40, color="orange", linestyle="--", linewidth=1, label="40% Min")
        ax6.fill_between(range(len(df)), df["Yield (%)"], alpha=0.1, color="#4361ee")
        ax6.set_xlabel("Prediction #")
        ax6.set_ylabel("Yield (%)")
        ax6.set_title("Yield Trend")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    else:
        st.info("No history yet! Run some predictions first.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — DRAW MODE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎨 Draw Mode — Sketch Defect Pattern")
    st.markdown("**Click/drag on the grid to mark defect dies → Get instant prediction!**")

    col_draw, col_draw_result = st.columns([1, 1])

    with col_draw:
        st.write("**Draw defect pattern (black = defect die):**")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#c8c8c8",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            draw_predict = st.button("🚀 Predict Drawing", use_container_width=True, type="primary")
        with col_b2:
            st.button("🗑️ Clear Canvas", use_container_width=True)

    with col_draw_result:
        if canvas_result.image_data is not None and draw_predict:
            # Convert canvas → PIL image
            canvas_array = canvas_result.image_data.astype(np.uint8)
            canvas_img   = PILImage.fromarray(canvas_array).convert("RGB")

            # Add circular wafer mask
            canvas_np  = np.array(canvas_img)
            h, w       = canvas_np.shape[:2]
            mask_circle = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_circle, (w // 2, h // 2), min(w, h) // 2 - 5, 255, -1)
            wafer_bg   = np.full_like(canvas_np, 200)
            result_img = np.where(mask_circle[:, :, None] > 0, canvas_np, 0)
            canvas_img = PILImage.fromarray(result_img.astype(np.uint8))

            with st.spinner("Predicting..."):
                result = predict(canvas_img, model, die_area)

            defect = result["defect_type"]
            conf   = result["confidence"]
            yld    = result["yield_pct"]
            dec    = result["decision"]

            badge_class = {"SAVE": "save-badge", "REVIEW": "review-badge", "SCRAP": "scrap-badge"}[dec]
            badge_emoji = {"SAVE": "✅", "REVIEW": "⚠️", "SCRAP": "❌"}[dec]

            st.write("**Prediction Result:**")
            st.markdown(f'<span class="{badge_class}">{badge_emoji} {dec}</span>', unsafe_allow_html=True)
            st.write("")

            d1, d2, d3 = st.columns(3)
            d1.metric("Defect",     defect)
            d2.metric("Confidence", f"{conf:.1f}%")
            d3.metric("Yield",      f"{yld}%")

            # Probs chart
            probs   = result["all_probs"]
            prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Prob"])
            prob_df = prob_df.sort_values("Prob", ascending=True)
            fig7, ax7 = plt.subplots(figsize=(5, 4))
            ax7.barh(prob_df["Class"], prob_df["Prob"],
                     color=["#4361ee" if c == defect else "#adb5bd" for c in prob_df["Class"]])
            ax7.set_xlabel("Probability (%)")
            ax7.set_title("Class Probabilities")
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()

            if groq_key:
                with st.spinner("Getting expert analysis..."):
                    explanation = get_explanation(defect, conf, yld, dec, groq_key)
                st.markdown(f'<div class="expert-box">{explanation.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Wafer ID":       wafer_id + "-DRAW",
                "Defect":         defect,
                "Confidence (%)": round(conf, 2),
                "Yield (%)":      yld,
                "Decision":       dec
            })
        else:
            st.info("Draw a defect pattern on the canvas and click Predict!")
