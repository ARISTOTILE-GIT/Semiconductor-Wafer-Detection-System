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
            df.style.map(color_decision, subset=["Decision"]),
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
    st.markdown("**Draw defect pattern on the wafer → Click Predict!**")

    col_draw, col_draw_result = st.columns([1, 1])

    with col_draw:
        canvas_html = """
        <style>
            #wrap { display:flex; flex-direction:column; align-items:center; gap:10px; font-family:sans-serif; }
            canvas { cursor:crosshair; image-rendering:pixelated; }
            .btns { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
            .btn  { padding:8px 18px; border:none; border-radius:8px; font-size:14px; font-weight:bold; cursor:pointer; }
            #clearBtn { background:#dc3545; color:white; }
            #undoBtn  { background:#6c757d; color:white; }
            #sendBtn  { background:#28a745; color:white; }
            #brushWrap { display:flex; align-items:center; gap:8px; font-size:13px; }
            #status { font-size:12px; color:#555; }
        </style>
        <div id="wrap">
            <div id="brushWrap">
                🖌️ Brush: <input type="range" id="brushSize" min="4" max="40" value="14" style="width:100px;">
                <span id="brushVal">14</span>px
            </div>
            <canvas id="wc" width="320" height="320"></canvas>
            <div class="btns">
                <button class="btn" id="undoBtn">↩️ Undo</button>
                <button class="btn" id="clearBtn">🗑️ Clear</button>
                <button class="btn" id="sendBtn">📤 Send to Predict</button>
            </div>
            <div id="status">Draw on the wafer, then click Send to Predict</div>
        </div>

        <script>
        const canvas = document.getElementById('wc');
        const ctx    = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const cx = W/2, cy = H/2, r = W/2 - 6;
        let painting = false;
        let brushSize = 14;
        let history   = [];

        document.getElementById('brushSize').addEventListener('input', function() {
            brushSize = parseInt(this.value);
            document.getElementById('brushVal').innerText = brushSize;
        });

        function drawBg() {
            ctx.clearRect(0,0,W,H);
            ctx.fillStyle = '#111';
            ctx.fillRect(0,0,W,H);
            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI*2);
            ctx.clip();
            ctx.fillStyle = '#c0c0c0';
            ctx.fillRect(0,0,W,H);
            ctx.restore();
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI*2);
            ctx.strokeStyle = '#888';
            ctx.lineWidth = 3;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy+r-2, 6, 0, Math.PI*2);
            ctx.fillStyle = '#111';
            ctx.fill();
        }

        function saveHistory() {
            history.push(canvas.toDataURL());
            if (history.length > 30) history.shift();
        }

        drawBg();
        saveHistory();

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const sx = W/rect.width, sy = H/rect.height;
            if (e.touches) return {
                x: (e.touches[0].clientX - rect.left)*sx,
                y: (e.touches[0].clientY - rect.top)*sy
            };
            return { x:(e.clientX-rect.left)*sx, y:(e.clientY-rect.top)*sy };
        }

        function inCircle(x,y) { return (x-cx)**2+(y-cy)**2 <= (r-4)**2; }

        function startDraw(e) {
            e.preventDefault();
            painting = true;
            saveHistory();
            const p = getPos(e);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
        }

        function endDraw(e) {
            e.preventDefault();
            painting = false;
            ctx.beginPath();
        }

        function draw(e) {
            e.preventDefault();
            if (!painting) return;
            const p = getPos(e);
            if (!inCircle(p.x, p.y)) return;
            ctx.lineWidth   = brushSize;
            ctx.lineCap     = 'round';
            ctx.lineJoin    = 'round';
            ctx.strokeStyle = '#111111';
            ctx.lineTo(p.x, p.y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
        }

        canvas.addEventListener('mousedown',  startDraw);
        canvas.addEventListener('mouseup',    endDraw);
        canvas.addEventListener('mouseleave', endDraw);
        canvas.addEventListener('mousemove',  draw);
        canvas.addEventListener('touchstart', startDraw, {passive:false});
        canvas.addEventListener('touchend',   endDraw,   {passive:false});
        canvas.addEventListener('touchmove',  draw,      {passive:false});

        document.getElementById('undoBtn').onclick = function() {
            if (history.length > 1) {
                history.pop();
                const img = new Image();
                img.src = history[history.length - 1];
                img.onload = () => ctx.drawImage(img, 0, 0);
                document.getElementById('status').innerText = 'Undone!';
            } else {
                drawBg();
                history = [];
                saveHistory();
                document.getElementById('status').innerText = 'Nothing to undo!';
            }
        };

        document.getElementById('clearBtn').onclick = function() {
            drawBg();
            history = [];
            saveHistory();
            document.getElementById('status').innerText = 'Cleared!';
        };

        document.getElementById('sendBtn').onclick = function() {
            const data = canvas.toDataURL('image/png');
            // Find all textareas in parent and set value
            const textareas = window.parent.document.querySelectorAll('textarea');
            let sent = false;
            textareas.forEach(ta => {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(ta, data);
                ta.dispatchEvent(new Event('input', {bubbles:true}));
                sent = true;
            });
            document.getElementById('status').innerText = sent
                ? '✅ Sent! Now click Predict Drawing below.'
                : '❌ Could not send. Try again.';
        };
        </script>
        """
        components.html(canvas_html, height=460)

        img_data     = st.text_area("img", key="draw_data", height=50, label_visibility="collapsed")
        draw_predict = st.button("🚀 Predict Drawing", use_container_width=True, type="primary")

    with col_draw_result:
        if draw_predict:
            raw = (img_data or "").strip()
            if raw and raw.startswith("data:image"):
                try:
                    _, encoded = raw.split(",", 1)
                    img_bytes  = base64.b64decode(encoded)
                    draw_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    st.image(draw_img, caption="Your drawing", width=220)

                    with st.spinner("Predicting..."):
                        result = predict(draw_img, model, die_area)

                    defect = result["defect_type"]
                    conf   = result["confidence"]
                    yld    = result["yield_pct"]
                    dec    = result["decision"]

                    badge_class = {"SAVE":"save-badge","REVIEW":"review-badge","SCRAP":"scrap-badge"}[dec]
                    badge_emoji = {"SAVE":"✅","REVIEW":"⚠️","SCRAP":"❌"}[dec]
                    st.markdown(f'<span class="{badge_class}">{badge_emoji} {dec}</span>', unsafe_allow_html=True)
                    st.write("")

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Defect",     defect)
                    d2.metric("Confidence", f"{conf:.1f}%")
                    d3.metric("Yield",      f"{yld}%")

                    probs   = result["all_probs"]
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Class","Prob"])
                    prob_df = prob_df.sort_values("Prob", ascending=True)
                    fig7, ax7 = plt.subplots(figsize=(5,4))
                    ax7.barh(prob_df["Class"], prob_df["Prob"],
                             color=["#4361ee" if c==defect else "#adb5bd" for c in prob_df["Class"]])
                    ax7.set_xlabel("Probability (%)"); ax7.set_title("Class Probabilities")
                    plt.tight_layout(); st.pyplot(fig7); plt.close()

                    if groq_key:
                        with st.spinner("Getting expert analysis..."):
                            explanation = get_explanation(defect, conf, yld, dec, groq_key)
                        st.markdown(f'<div class="expert-box">{explanation.replace(chr(10),"<br>")}</div>',
                                    unsafe_allow_html=True)

                    st.session_state.history.append({
                        "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Wafer ID":       wafer_id + "-DRAW",
                        "Defect":         defect,
                        "Confidence (%)": round(conf, 2),
                        "Yield (%)":      yld,
                        "Decision":       dec
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Click **📤 Send to Predict** on canvas first, then click Predict Drawing!")
        else:
            st.info("""
            **Steps:**
            1. 🎨 Draw defect pattern on wafer
            2. Click **📤 Send to Predict** button
            3. Click **🚀 Predict Drawing** below
            """)
