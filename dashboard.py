# dashboard.py — Redesigned AI-IDS Streamlit dashboard (updated & fixed)
# Drop this file into your app folder and run: streamlit run dashboard.py

import os
import json
import tempfile
import joblib
from typing import Optional, List

import pandas as pd
import streamlit as st

from pcap_feature_extractor import extract_features_from_pcap

# ---------- Page config ----------
st.set_page_config(page_title="AI Intrusion Detection System", page_icon="🛡️", layout="wide")

# ---------- Paths & constants ----------
MODEL_PATH = os.path.join("models", "ids_model.pkl")
DEFAULT_PCAP = os.path.join("sample_pcaps", "2026-02-28-traffic-analysis-exercise.pcap")
EXPECTED_FEATURES = 42
MAX_TOP_ROWS = 500

# ---------- Helper: safe cache clear ----------
def clear_model_cache():
    """
    Try various cache clearing approaches so we don't crash on Streamlit versions
    that don't have certain APIs. Also clear common session_state keys used
    by previous runs.
    """
    # Remove common session_state keys we might have used
    for key in ["loaded_model", "model_cached", "model"]:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception:
                pass

    # Try modern cache clearing APIs but don't crash if they aren't present
    try:
        # recommended for resources (Streamlit >= 1.18-ish)
        st.cache_resource.clear()
    except Exception:
        try:
            # older cache API
            st.experimental_memo.clear()
        except Exception:
            try:
                st.cache_data.clear()
            except Exception:
                # last resort: do nothing — we've already cleared session_state keys
                pass

# ---------- CSS & themes ----------
_COMMON_CSS = r"""
<style>
:root{
  --accent1: #0b76ff;
  --accent2: #00b894;
  --card-bg: rgba(255,255,255,0.02);
  --muted-dark: #9fb4d8;
  --muted-light: #475569;
}

/* Layout containers */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 18px; }
.summary-card { border-radius: 12px; padding: 12px; position: relative; overflow: hidden; min-height: 88px; }
.summary-card .title { font-size: 0.9rem; font-weight: 700; margin-bottom: 6px; }
.summary-card .value { font-size: 1.55rem; font-weight: 800; }
.summary-card .sub { font-size: 0.82rem; color: var(--muted, rgba(0,0,0,0.45)); }

.kpi-row { display:flex; gap:10px; align-items:center; }
.kpi-small { font-size:0.9rem; color:var(--muted); }

/* small responsive tweaks */
@media (max-width:880px){ .card-grid { grid-template-columns: 1fr; } }
</style>
"""

# theme css fragments
_DARK_CSS = """
<style>
:root{ --muted: #9fb4d8; --card-bg: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); }
.stApp { background: linear-gradient(180deg,#041022,#07182a) !important; color: #e6f0ff !important; }
.summary-card { background: var(--card-bg); box-shadow: 0 8px 30px rgba(2,6,23,0.6); color: #e6f0ff; }
</style>
"""

_LIGHT_CSS = """
<style>
:root{ --muted: #475569; --card-bg: linear-gradient(180deg,#ffffff,#f7f9fb); }
.stApp { background: linear-gradient(180deg,#f8fafc,#eef2ff) !important; color: #071023 !important; }
.summary-card { background: var(--card-bg); box-shadow: 0 6px 18px rgba(16,24,40,0.06); color: #071023; }
</style>
"""

_PRO_BG_CSS = """
<style>
/* subtle textured professional background - accessible contrast */
body > div[role="application"] {
  background-image:
    radial-gradient(circle at 10% 10%, rgba(11,118,255,0.03), transparent 10%),
    linear-gradient(180deg, rgba(3,10,20,0.6), rgba(3,10,20,0.85));
  background-attachment: fixed;
}
</style>
"""

# ---------- Model loading with caching ----------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # joblib.load may return a scikit-learn-ish model or a custom object
    return joblib.load(path)

# ---------- Prediction helpers ----------
def safe_predict(model, X: pd.DataFrame):
    """
    Return (probs:list[float], preds_int:list[int]).
    Handles models without predict_proba gracefully.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_proc = X.copy().fillna(0)

    # attempt predict_proba -> handle binary / single-dim outputs
    try:
        proba = model.predict_proba(X_proc)
        # proba shape handling: if shape is (n,) or (n,1) handle carefully
        if hasattr(proba, "ndim") and proba.ndim == 1:
            probs = [float(p) for p in proba]
        else:
            # standard sklearn: choose column 1 as 'positive' class
            try:
                probs = [float(p) for p in proba[:, 1]]
            except Exception:
                # fallback to first col if only one column present
                probs = [float(p) for p in proba[:, 0]]
    except Exception:
        # fallback to predict -> turn into 0.0/1.0 probabilities
        try:
            preds = model.predict(X_proc)
            probs = [1.0 if int(p) == 1 else 0.0 for p in preds]
        except Exception:
            probs = [0.0] * len(X_proc)

    preds_int = [1 if (p is not None and p >= 0.5) else 0 for p in probs]
    return probs, preds_int

def pad_features(df: pd.DataFrame, expected_cols: int = EXPECTED_FEATURES) -> pd.DataFrame:
    """
    Ensures dataframe has at least expected_cols columns. Adds deterministic dummy names.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    cur = df.shape[1]
    if cur < expected_cols:
        for i in range(expected_cols - cur):
            col_name = f"dummy_{i}"
            # only add column if not present
            if col_name not in df.columns:
                df[col_name] = 0
    # Ensure column order deterministic
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def map_severity_by_prob(prob: float) -> str:
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.4:
        return "LOW"
    return "NORMAL"

# ---------- Sidebar (controls & theme) ----------
with st.sidebar:
    st.title("⚙️ Controls")
    ui_theme = st.selectbox("UI Theme", ["Soft-Dark (recommended)", "Light (high-contrast)"], index=0)
    background_style = st.selectbox("Background", ["Professional gradient (recommended)", "Plain"], index=0)

    st.markdown("---")
    label_preset = st.selectbox("Label Preset",
                                ["Binary (BENIGN / MALICIOUS)", "Severity (NORMAL/LOW/MEDIUM/HIGH)", "Custom labels"],
                                index=0)
    custom_label_0 = None
    custom_label_1 = None
    if label_preset == "Custom labels":
        custom_label_0 = st.text_input("Label for prediction 0", value="BENIGN")
        custom_label_1 = st.text_input("Label for prediction 1", value="MALICIOUS")

    threshold = st.slider("Probability threshold (suspicious)", 0.0, 1.0, 0.60, 0.01)
    prob_filter = st.slider("Show flows with prob >=", 0.0, 1.0, 0.0, 0.01)
    top_n = int(st.number_input("Top N suspicious flows", min_value=5, max_value=MAX_TOP_ROWS, value=20))
    show_raw = st.checkbox("Show raw features (large)", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload", value=True)

    st.markdown("---")
    if st.button("Clear cached model"):
        # Safe clear (no AttributeError risk)
        clear_model_cache()
        # ensure model variable is removed from session and rerun
        st.experimental_rerun()

# ---------- Inject CSS per theme ----------
st.markdown(_COMMON_CSS, unsafe_allow_html=True)
if ui_theme.startswith("Soft"):
    st.markdown(_DARK_CSS + (_PRO_BG_CSS if background_style.startswith("Professional") else ""), unsafe_allow_html=True)
else:
    st.markdown(_LIGHT_CSS + ("" if background_style == "Plain" else _PRO_BG_CSS), unsafe_allow_html=True)

# ---------- Load model ----------
try:
    model = load_model()
except FileNotFoundError:
    st.sidebar.error(f"Model missing at {MODEL_PATH}. Upload or place the model file in 'models/'.")
    model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# ---------- Header / Upload area ----------
st.markdown("# 🛡️ AI-Powered Intrusion Detection System")
st.markdown("Upload a PCAP, choose label scheme and run detection. The dashboard focuses on readable, accessible statistics.")

left, right = st.columns([2, 1])
with left:
    st.subheader("1) Upload PCAP")
    uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])
    sample_choice = st.selectbox("Or choose sample", ["Default sample (recommended)", "No sample / upload only"])
    pcap_path = None
    if uploaded is not None:
        # write to a temp file safely
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        pcap_path = tmp.name
    else:
        if sample_choice.startswith("Default"):
            if os.path.exists(DEFAULT_PCAP):
                pcap_path = DEFAULT_PCAP
            else:
                st.warning("Default sample PCAP not found on server.")
                pcap_path = None

    st.markdown("**PCAP:**")
    st.write(pcap_path if pcap_path else "No PCAP selected")

with right:
    st.subheader("Model & Quick Info")
    model_info_html = f"""
    <div class="summary-card" style="padding:12px">
      <div class="title">Model Path</div>
      <div class="value" title="{MODEL_PATH}">{os.path.basename(MODEL_PATH)}</div>
      <div class="sub">Threshold: {threshold:.2f}</div>
    </div>
    """
    st.markdown(model_info_html, unsafe_allow_html=True)

# ---------- Run logic ----------
do_run = (run_on_upload and pcap_path is not None) or st.button("Extract & Predict ▶️")

if not pcap_path:
    st.info("No PCAP selected — upload or choose sample to run detection.")

# ---------- When running: extract, predict, show visuals ----------
if do_run and pcap_path and model is not None:
    with st.spinner("Extracting features from PCAP... ⛏️"):
        try:
            df = extract_features_from_pcap(pcap_path)
            if df is None or df.shape[0] == 0:
                st.warning("No flows parsed from PCAP. Check the file content.")
                st.stop()
        except Exception as e:
            st.exception(f"Feature extraction failed: {e}")
            st.stop()

    df = pad_features(df, EXPECTED_FEATURES)

    with st.spinner("Running model inference... 🤖"):
        probs, preds_int = safe_predict(model, df)

    results = df.copy().reset_index(drop=True)
    results["malicious_probability"] = [float(p) for p in probs]
    results["prediction_int"] = [int(x) for x in preds_int]
    results["is_suspicious"] = results["malicious_probability"] >= threshold

    # labels
    if label_preset == "Binary (BENIGN / MALICIOUS)":
        lbl0, lbl1 = "BENIGN", "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})
    elif label_preset == "Severity (NORMAL/LOW/MEDIUM/HIGH)":
        results["label"] = results["malicious_probability"].apply(map_severity_by_prob)
    else:
        lbl0 = custom_label_0 if custom_label_0 else "BENIGN"
        lbl1 = custom_label_1 if custom_label_1 else "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})

    total = len(results)
    attacks = int(results["prediction_int"].sum())
    normals = total - attacks
    attack_ratio = attacks / total if total else 0.0

    # threat level
    if attack_ratio < 0.05:
        threat = "VERY LOW"
        color_emoji = "🟢"
    elif attack_ratio < 0.15:
        threat = "LOW"
        color_emoji = "🟢"
    elif attack_ratio < 0.3:
        threat = "MEDIUM"
        color_emoji = "🟠"
    else:
        threat = "HIGH"
        color_emoji = "🔴"

    # ---- Summary KPI cards ----
    st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
    kpis = [
        ("Total Flows", total, "Total parsed flows"),
        ("Benign / Normal", normals, "Non-alert flows"),
        ("Suspicious / Alerts", attacks, "Detected suspicious flows"),
        ("Threat Level", f"{threat} {color_emoji}", f"{attack_ratio*100:.2f}% of flows"),
    ]
    for title, value, sub in kpis:
        st.markdown(
            f"""
            <div class='summary-card'>
              <div class='title'>{title}</div>
              <div class='value'>{value}</div>
              <div class='sub'>{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Traffic overview charts ----
    st.markdown("## Traffic Overview")
    try:
        import altair as alt
        # label distribution bar chart
        label_counts = results.groupby("label").size().reset_index(name="count").sort_values("count", ascending=False)
        bar = (
            alt.Chart(label_counts)
            .mark_bar()
            .encode(x=alt.X("label:N", title="Label", sort="-y"), y=alt.Y("count:Q", title="Count"), tooltip=["label", "count"])
            .properties(height=220)
        )
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        st.bar_chart(results["label"].value_counts())

    # Probability distribution / histogram
    st.markdown("## Probability Distribution")
    try:
        import altair as alt
        hist = (
            alt.Chart(results)
            .mark_bar()
            .encode(
                x=alt.X("malicious_probability:Q", bin=alt.Bin(maxbins=40), title="Malicious probability"),
                y=alt.Y("count():Q", title="Flows"),
                tooltip=[alt.Tooltip("count()", title="Flows")]
            )
            .properties(height=220)
        )
        st.altair_chart(hist, use_container_width=True)
    except Exception:
        st.bar_chart(results["malicious_probability"].value_counts().sort_index())

    # attack probability over sample
    st.markdown("## Attack Probability Over Sample (first 200 rows)")
    st.line_chart(results["malicious_probability"].head(200))

    # ---- Interactive table and top suspicious flows ----
    st.markdown("## Detected Flows — Interactive View")
    filtered = results[results["malicious_probability"] >= prob_filter].sort_values("malicious_probability", ascending=False)
    if filtered.shape[0] == 0:
        st.info("No flows match the current probability filter. Lower the filter or choose a different PCAP.")
    else:
        top = filtered.head(top_n).copy()
        # Present a clean set of columns
        display_cols = ["label", "malicious_probability", "prediction_int"] + [c for c in top.columns if c.startswith("dummy_")][:3]
        display_cols = [c for c in display_cols if c in top.columns]
        display_df = top[display_cols]

        # highlight suspicious rows using Styler if available
        if highlight_suspicious:
            try:
                styled = display_df.style.apply(lambda row: ["background-color: rgba(255,0,0,0.10)" if row["malicious_probability"] >= threshold else "" for _ in row], axis=1)
                st.dataframe(styled, height=400)
            except Exception:
                st.dataframe(display_df, height=400)
        else:
            st.dataframe(display_df, height=400)

    # ---- Inspect single flow / JSON view ----
    st.markdown("## Inspect a single flow / JSON view")
    idx = st.number_input("Row index (0-based)", 0, max(0, total - 1), 0)
    if total > 0:
        row = results.iloc[int(idx)].to_dict()
        st.json(row)

    # ---- Export ----
    st.markdown("## Export results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    json_str = results.to_json(orient="records", indent=2)
    st.download_button("📥 Download CSV", csv_bytes, file_name="ai_ids_results.csv", mime="text/csv")
    st.download_button("📥 Download JSON", json_str.encode("utf-8"), file_name="ai_ids_results.json", mime="application/json")

    if st.button("💾 Save results to predictions.csv (server)"):
        out_path = "predictions.csv"
        results.to_csv(out_path, index=False)
        st.success(f"Saved to {out_path}")

    if show_raw:
        st.markdown("---")
        st.subheader("Raw extracted features (first 300 rows)")
        st.dataframe(df.head(300))

    st.success("Detection completed ✅")

else:
    st.markdown("<div class='summary-card' style='padding:14px'>", unsafe_allow_html=True)
    st.markdown("## Welcome 👋")
    st.markdown(
        "This dashboard extracts features from PCAPs and runs a trained ML model to detect suspicious activity.\n\nUse the controls on the left to configure labels, thresholds and run detection.")
    st.markdown("</div>", unsafe_allow_html=True)

# close message
st.markdown("---")
st.markdown("Made with ❤️ by the AI-IDS team")