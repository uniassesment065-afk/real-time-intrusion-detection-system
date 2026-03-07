"""
dashboard.py — Improved AI-IDS Streamlit dashboard
Features added:
- Label presets dropdown (Binary, Severity, Custom)
- Custom label inputs
- Dark-theme text/metric fixes (high contrast)
- Probability filter + top-N suspicious flows
- Better UX (spinners, cached model load, safe predict)
- Export JSON & CSV, copy JSON to clipboard
Note: keeps using your pcap_feature_extractor.extract_features_from_pcap
"""

import os
import streamlit as st
import pandas as pd
import joblib
import json
from typing import Optional

from pcap_feature_extractor import extract_features_from_pcap

# ---------- Config ----------
st.set_page_config(
    page_title="AI Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join("models", "ids_model.pkl")
DEFAULT_PCAP = os.path.join("sample_pcaps", "2026-02-28-traffic-analysis-exercise.pcap")
EXPECTED_FEATURES = 42  # keep this as your model expects

# ---------- Styles (make text readable on dark background) ----------
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(90deg,#081224 0%, #041022 100%);
        color: #e6eef8;
    }

    /* card look */
    .card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    }

    /* ensure headings and general text are light */
    .stMarkdown, .stText, .stHeader, .stSubheader, .stMetric {
        color: #e6eef8 !important;
    }

    /* metric label & value color adjustments */
    .stMetric .metric-label, .stMetric .metric-value {
        color: #e6eef8 !important;
    }

    /* dataframe: header and cells */
    div[data-testid="stDataFrameContainer"] table thead th {
        color: #e6eef8 !important;
        background: rgba(255,255,255,0.03) !important;
    }
    div[data-testid="stDataFrameContainer"] table tbody td {
        color: #e6eef8 !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* buttons */
    .stButton>button {
        background-image: linear-gradient(90deg,#6366f1,#06b6d4) !important;
        color: white !important;
        border: none !important;
    }

    /* small footers */
    .footer {
        color: #9aaed6;
        font-size: 0.9em;
        margin-top: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    """Load and cache the model. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def safe_predict(model, X: pd.DataFrame) -> (list, list):
    """
    Return (probs, preds_int):
    - probs: list of float probabilities (0..1) when possible, otherwise fallback to ints
    - preds_int: 0/1 ints based on probs or direct predict
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Ensure columns numeric/filled if necessary (simple safety)
    X_proc = X.copy().fillna(0)

    # try predict_proba
    try:
        proba = model.predict_proba(X_proc)
        # some estimators return shape (n_samples, ) or (n_samples, n_classes)
        if proba.ndim == 1:
            probs = [float(p) for p in proba]
        else:
            probs = [float(p) for p in proba[:, 1]]
    except Exception:
        # fallback: model.predict -> treat 1 as high confidence, 0 as low
        try:
            preds = model.predict(X_proc)
            probs = [1.0 if int(p) == 1 else 0.0 for p in preds]
        except Exception as e:
            # last resort: mark everything unknown (0.0)
            probs = [0.0] * len(X_proc)

    preds_int = [1 if (p is not None and p >= 0.5) else 0 for p in probs]
    return probs, preds_int


def pad_features(df: pd.DataFrame, expected_cols: int = EXPECTED_FEATURES) -> pd.DataFrame:
    """Pad dataframe with zero-valued dummy columns to match expected feature count."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    cur = df.shape[1]
    if cur < expected_cols:
        # add deterministic dummy names starting after existing columns
        for i in range(expected_cols - cur):
            df[f"dummy_{i}"] = 0
    return df


def map_severity_by_prob(prob: float) -> str:
    """Map probability to severity label (used in 'Severity' preset)."""
    # thresholds are adjustable; user can edit or later expose as settings
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.4:
        return "LOW"
    return "NORMAL"


# ---------- Sidebar controls ----------
with st.sidebar:
    st.title("⚙️ Controls")

    st.markdown("**Label Preset**")
    label_preset = st.selectbox(
        "Choose a label preset",
        options=["Binary (BENIGN / MALICIOUS)", "Severity (NORMAL/LOW/MEDIUM/HIGH)", "Custom labels"]
    )

    # if custom, let user set label(s)
    if label_preset == "Custom labels":
        st.markdown("Enter label(s) for 0 and 1 predictions:")
        custom_label_0 = st.text_input("Label for prediction 0", value="BENIGN")
        custom_label_1 = st.text_input("Label for prediction 1", value="MALICIOUS")
    else:
        custom_label_0 = None
        custom_label_1 = None

    # user may want to pick from common label pairs quickly (dropdown)
    st.markdown("Or pick a quick pair:")
    quick_pair = st.selectbox("Common pairs", options=["(BENIGN, MALICIOUS)", "(NORMAL, ATTACK)", "(GOOD, BAD)", "— none —"], index=0)
    if quick_pair != "— none —" and label_preset == "Custom labels":
        left, right = quick_pair.strip("()").split(",")
        # only populate if custom preset chosen
        custom_label_0 = custom_label_0 if custom_label_0 else left.strip()
        custom_label_1 = custom_label_1 if custom_label_1 else right.strip()

    # Threshold
    threshold = st.slider("Probability threshold to treat as 'suspicious' (for metrics)", 0.0, 1.0, 0.60, 0.01)

    # Probability filter for table view
    prob_filter = st.slider("Show only flows with probability >= ", 0.0, 1.0, 0.0, 0.01)

    # top N suspicious rows to show
    top_n = st.number_input("Top N suspicious flows to show", min_value=5, max_value=200, value=20, step=5)

    show_raw = st.checkbox("Show raw extracted features (large tables)", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows (table)", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload / sample", value=True)

    st.markdown("---")
    if st.button("Run Detection Now 🔎"):
        st.session_state["force_run"] = True

    if st.button("Clear cached model"):
        try:
            # For Streamlit versions: evict cached resource
            st.cache_resource.clear()
        except Exception:
            # fallback: remove session var
            if "loaded_model" in st.session_state:
                del st.session_state["loaded_model"]
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("App built with ❤️ — tweak settings to experiment.")


# ---------- Load model ----------
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(f"Model missing at {MODEL_PATH}. Please upload or place the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------- Main layout ----------
st.title("🛡️ AI-Powered Intrusion Detection System")
st.markdown("Upload a PCAP, pick label scheme, and run the detection. Use the probability filter and top-N view to inspect high-risk flows.")

cols = st.columns([2, 1])
with cols[0]:
    st.subheader("1) Upload PCAP (or use default sample)")
    uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])
    sample_choice = st.selectbox("Or choose a sample", options=["Default sample (recommended)", "No sample / upload only"], index=0)

    if uploaded is not None:
        pcap_path = "temp_uploaded.pcap"
        with open(pcap_path, "wb") as f:
            f.write(uploaded.read())
    else:
        pcap_path = DEFAULT_PCAP if sample_choice.startswith("Default") else None

    st.markdown("**PCAP:**")
    st.write(pcap_path if pcap_path else "No PCAP selected")

with cols[1]:
    st.subheader("Model & Quick Info")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"- Model path: `{MODEL_PATH}`")
    st.markdown(f"- Threshold (display): **{threshold:.2f}**")
    st.markdown("</div>", unsafe_allow_html=True)

# run toggle logic
do_run = st.session_state.get("force_run", False) or (run_on_upload and pcap_path is not None) or st.button("Extract & Predict ▶️")
st.session_state["force_run"] = False

if not pcap_path:
    st.info("No PCAP selected — upload a PCAP or choose the default sample to run detection.")

# ---------- Feature extraction & prediction ----------
if do_run and pcap_path:
    with st.spinner("Extracting features from PCAP... ⛏️"):
        try:
            df = extract_features_from_pcap(pcap_path)
            if df is None or df.shape[0] == 0:
                st.warning("No flows/packets parsed from the PCAP. Check the file content.")
                st.stop()
        except Exception as e:
            st.exception(f"Feature extraction failed: {e}")
            st.stop()

    # pad features to expected count
    df = pad_features(df, EXPECTED_FEATURES)

    # Run safe prediction
    with st.spinner("Running model inference... 🤖"):
        probs, preds_int = safe_predict(model, df)

    # add inference columns
    results = df.copy().reset_index(drop=True)
    results["malicious_probability"] = [float(p) for p in probs]
    results["prediction_int"] = [int(x) for x in preds_int]
    results["is_suspicious"] = results["malicious_probability"] >= threshold

    # Map labels depending on preset
    if label_preset == "Binary (BENIGN / MALICIOUS)":
        lbl0, lbl1 = "BENIGN", "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})
    elif label_preset == "Severity (NORMAL/LOW/MEDIUM/HIGH)":
        # severity determined by probability, independent of raw prediction (useful to grade risk)
        results["label"] = results["malicious_probability"].apply(map_severity_by_prob)
    else:  # custom labels
        # use user-specified or fallback to defaults
        lbl0 = custom_label_0 if custom_label_0 else "BENIGN"
        lbl1 = custom_label_1 if custom_label_1 else "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})

    # basic metrics
    total = len(results)
    attacks = int(results["prediction_int"].sum())
    normals = total - attacks
    attack_ratio = attacks / total if total else 0.0

    # threat level (simple heuristics)
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

    # Top summary cards
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flows", total, "")
    c2.metric("Benign / Normal", normals, "")
    c3.metric("Suspicious / Alerts", attacks, "")
    c4.metric("Threat Level", f"{threat} {color_emoji}", f"{attack_ratio*100:.2f}% of flows")
    st.markdown("</div>", unsafe_allow_html=True)

    # Visualization: Distribution and probability histogram
    st.subheader("Traffic Overview")
    dist_df = results.groupby("label").size().reset_index(name="count").sort_values("count", ascending=False)
    st.bar_chart(dist_df.set_index("label"))

    st.subheader("Probability Distribution (histogram)")
    st.altair_chart(
        # avoid extra dependency on seaborn; use Altair (already in many Streamlit environments)
        __import__("altair").Chart(results).mark_bar().encode(
            x=__import__("altair").X("malicious_probability:Q", bin=__import__("altair").Bin(maxbins=30)),
            y="count()"
        ).properties(height=200),
        use_container_width=True
    )

    # Probability line / sample sequence
    st.subheader("Attack Probability Over Sample (first 100 rows)")
    st.line_chart(results["malicious_probability"].head(100))

    # Table with filtering
    st.subheader("Detected Flows — Interactive View")
    filtered = results[results["malicious_probability"] >= prob_filter].sort_values("malicious_probability", ascending=False)
    if filtered.shape[0] == 0:
        st.info("No flows match the current probability filter. Lower the filter or choose a different PCAP.")
    else:
        # show top N suspicious flows (by probability)
        top = filtered.head(top_n).copy()
        # highlight suspicious rows visually using pandas styler if requested
        if highlight_suspicious:
            def _style_row(row):
                bg = "rgba(255,0,0,0.12)" if row["is_suspicious"] else ""
                return ["background: " + bg if col in top.columns else "" for col in row.index]
            try:
                st.dataframe(top.style.apply(lambda r: ["background-color: rgba(255,0,0,0.12)" if r["is_suspicious"] else "" for _ in r], axis=1), height=360)
            except Exception:
                st.dataframe(top, height=360)
        else:
            st.dataframe(top[["label", "malicious_probability", "prediction_int"] + ([c for c in top.columns if c.startswith("dummy_")][:3])], height=360)

    # Inspect a single flow
    st.subheader("Inspect a single flow / JSON view")
    idx = st.number_input("Row index (0-based)", 0, max(0, total - 1), 0)
    if total > 0:
        row = results.iloc[int(idx)].to_dict()
        st.json(row)

    # Export: CSV & JSON & copyable JSON
    st.subheader("Export results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    json_str = results.to_json(orient="records", indent=2)
    st.download_button("📥 Download CSV", csv_bytes, file_name="ai_ids_results.csv", mime="text/csv")
    st.download_button("📥 Download JSON", json_str.encode("utf-8"), file_name="ai_ids_results.json", mime="application/json")
    if st.button("Copy JSON to clipboard (browser)"):
        # streamlit can't directly access clipboard on server; provide JSON in a text area for manual copy
        st.text_area("Copy the JSON below", value=json_str, height=260)

    # Optionally save locally on the server (useful for quick dev)
    if st.button("💾 Save results to predictions.csv (server)"):
        out_path = "predictions.csv"
        results.to_csv(out_path, index=False)
        st.success(f"Saved to {out_path}")

    st.success("Detection completed ✅")

else:
    # Welcome + demo controls
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Welcome 👋")
    st.markdown(
        "This dashboard extracts features from PCAPs and runs a trained ML model to detect suspicious activity.\n\n"
        "Use the **Label Preset** to choose how results are labeled (binary, severity by probability, or custom). "
        "Upload a PCAP or use the default sample and click **Run Detection**."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ by the AI-IDS team — replace text & model as needed.</div>", unsafe_allow_html=True)