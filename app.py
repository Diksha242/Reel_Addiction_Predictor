import streamlit as st
import numpy as np
import joblib
import os

# ---------------------------
# Paths (adjust if needed)
# ---------------------------
MODEL_PATH = os.path.join("models", "final_svm_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
COLUMNS_PATH = os.path.join("models", "columns.pkl")

# <-- use this uploaded image as reference / decoration -->
REFERENCE_IMAGE = "/mnt/data/838ec1cb-4f5d-4022-bfc8-b9a7f56cbf5d.png"

# ---------------------------
# Load artifacts (safe)
# ---------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

model = safe_load(MODEL_PATH)
scaler = safe_load(SCALER_PATH)
columns = safe_load(COLUMNS_PATH)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Reel Addiction Predictor", page_icon="🎞", layout="centered")

# ---------------------------
# Theme toggle (sidebar)
# ---------------------------
mode = st.sidebar.radio("Theme Mode", ["🌙 Dark", "🔆 Light"])
is_dark = mode == "🌙 Dark"

bg = "#0f1217" if is_dark else "#ffffff"
fg = "#ffffff" if is_dark else "#111827"
card_bg = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.03)"
muted = "#94a3b8" if is_dark else "#94a3b8"

# ---------------------------
# CSS (safe targeting)
# ---------------------------
st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: {bg}; color: {fg}; }}
[data-testid="stSidebar"] {{ background-color: {bg}; color: {fg}; }}
.card {{
  background: {card_bg};
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 14px;
  border: 1px solid rgba(255,255,255,0.03);
}}
h1 {{ color: {fg}; font-size: 36px; }}
.small-muted {{ color: {muted}; font-size:13px; text-align:center; }}
.gauge-wrap {{ max-width:720px; margin-left:auto; margin-right:auto; }}
@media (max-width:600px) {{ .gauge-wrap {{ max-width:340px; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
    <div style="text-align:center; margin-bottom:8px;">
      <h1>🎞 Reel Addiction Predictor</h1>
      <div class="small-muted">Enter user activity and press Predict — speedometer shows risk</div>
    </div>
""", unsafe_allow_html=True)

# show the uploaded reference image (optional decoration)
if os.path.exists(REFERENCE_IMAGE):
    st.image(REFERENCE_IMAGE, use_column_width=False, width=560)

# ---------------------------
# Input card
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📌 Enter user activity details")
c1, c2 = st.columns(2)
with c1:
    daily_usage = st.number_input("Daily Usage (hours)", 0.0, 24.0, 1.0, step=0.1)
    avg_screen = st.number_input("Average Screen Time (minutes)", 0, 600, 60)
    likes = st.number_input("Likes per day", 0, 5000, 30)
with c2:
    comments = st.number_input("Comments per day", 0, 2000, 10)
    shares = st.number_input("Shares per day", 0, 2000, 5)
    videos = st.number_input("Videos watched per day", 0, 3000, 40)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Predict
# ---------------------------
predict = st.button("🎯 Predict Addiction Level", use_container_width=True)

if predict:
    if (daily_usage == 0 and avg_screen == 0 and likes == 0 and comments == 0):
        st.error("Please enter some activity values before predicting.")
    elif model is None or scaler is None:
        st.error("Model or scaler not loaded. Check models/ folder.")
    else:
        # prepare feature vector, scale, predict
        X = np.array([[daily_usage, avg_screen, likes, comments, shares, videos]])
        try:
            Xs = scaler.transform(X)
            pred = int(model.predict(Xs)[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            pred = 0

        labels = ["Low Risk", "Moderate Risk", "High Risk"]
        label = labels[pred] if pred < len(labels) else "Unknown"

        # map to degrees: left (-90) low, center (0) moderate, right (+90) high
        deg_map = {0: -90, 1: 0, 2: 90}
        target_deg = deg_map.get(pred, 0)

        score_pct = int((pred / 2.0) * 100)

        # show textual result
        st.markdown(f"<div style='text-align:center; margin-top:10px;'><strong style='color:{fg}; font-size:18px'>Prediction: {label}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; color:{muted}; margin-bottom:8px'>Score: {score_pct}%</div>", unsafe_allow_html=True)

        # ---------------------------
        # Speedometer SVG + JS (matches uploaded visual)
        # ---------------------------
        # Insert the numeric target_deg directly into JS (no template literals)
        gauge_html = """
<div class="gauge-wrap">
  <svg viewBox="0 0 800 440" preserveAspectRatio="xMidYMid meet" style="width:100%; height:auto;">
    <defs>
      <linearGradient id="segGrad" x1="0%" x2="100%">
        <stop offset="0%" stop-color="#3CCF7A"/>    <!-- green -->
        <stop offset="0.5" stop-color="#FBE36B"/>  <!-- yellow -->
        <stop offset="1" stop-color="#F25A4A"/>    <!-- red -->
      </linearGradient>
      <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
        <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000" flood-opacity="0.18"/>
      </filter>
    </defs>

    <!-- thick colored arc -->
    <path d="M 80 360 A 320 320 0 0 1 720 360" fill="none" stroke="url(#segGrad)" stroke-width="56" stroke-linecap="round" />

    <!-- outer rim -->
    <path d="M 60 360 A 340 340 0 0 1 740 360" fill="none" stroke="#2F3840" stroke-width="18" stroke-linecap="round" />

    <!-- ticks (major) -->
    <g stroke="#2F3840" stroke-width="8" stroke-linecap="round" opacity="0.9">
      <line x1="110" y1="345" x2="130" y2="315" />
      <line x1="230" y1="260" x2="250" y2="235" />
      <line x1="400" y1="190" x2="400" y2="160" />
      <line x1="550" y1="235" x2="570" y2="260" />
      <line x1="690" y1="345" x2="670" y2="315" />
    </g>

    <!-- labels -->
    <text x="150" y="320" font-size="36" fill="#2F3840" font-family="Arial">Low</text>
    <text x="400" y="110" font-size="36" fill="#2F3840" text-anchor="middle" font-family="Arial">Moderate</text>
    <text x="650" y="320" font-size="36" fill="#2F3840" font-family="Arial">High</text>

    <!-- hub -->
    <circle cx="400" cy="360" r="28" fill="#2F3840" />

    <!-- needle group (rotate around hub) -->
    <g id="needle" transform="translate(400,360) rotate(-90)">
      <!-- long needle shape -->
      <path d="M 0 -10 L 6 -8 L 40 -120 L 20 -120 L -6 -30 Z" fill="#2F3840" />
      <!-- center disc -->
      <circle cx="0" cy="0" r="18" fill="#2F3840" />
    </g>
  </svg>

  <div style="text-align:center; margin-top:18px;">
    <div style="font-weight:700; font-size:34px; color:__FG__;">Prediction: __LABEL__</div>
    <div style="font-size:28px; color:__MUTED__; margin-top:6px;">Score: __SCORE__%</div>
  </div>
</div>

<script>
(function(){
  const needle = document.getElementById('needle');
  const start = performance.now();
  const duration = 900;
  const startDeg = -90;
  const targetDeg = __TARGET__;

  function animate(now){
    const t = Math.min(1, (now - start) / duration);
    const ease = 1 - Math.pow(1 - t, 3);
    const current = startDeg + (targetDeg - startDeg) * ease;
    needle.setAttribute('transform', 'translate(400,360) rotate(' + current + ')');
    if (t < 1) requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
</script>
"""
        # Replace placeholders safely (avoid f-string brace conflicts)
        gauge_html = gauge_html.replace("__LABEL__", label).replace("__SCORE__", str(score_pct)).replace("__FG__", fg).replace("__MUTED__", muted).replace("__TARGET__", str(target_deg))
        st.components.v1.html(gauge_html, height=480, scrolling=False)
