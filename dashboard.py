# dashboard.py
"""
AI-Powered Intrusion Detection — Interactive Streamlit Dashboard
Add this file to your repo (replace your current dashboard file).
"""

import os
import io
import time
import json
import base64
from typing import List, Optional

import streamlit as st
import pandas as pd
import joblib

from pcap_feature_extractor import extract_features_from_pcap  # keep using your extractor

# ---------- Page config and basic styling ----------
st.set_page_config(
    page_title="AI Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for nicer visuals
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(90deg, #0f172a 0%, #071634 40%, #041022 100%);
        color: #e6eef8;
    }
    /* Card style for sections */
    .card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(2,6,23,0.6);
        margin-bottom: 16px;
    }
    /* Styled buttons */
    .stButton>button {
        background-image: linear-gradient(90deg,#6366f1,#06b6d4);
        color: white;
        border: none;
    }
    /* Smaller footer text */
    .footer {
        color: #9aaed6;
        font-size: 0.9em;
        margin-top: 12px;
    }
    /* Make dataframe container have rounded corners */
    div[data-testid="stDataFrameContainer"] > div {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper utilities ----------

MODEL_PATH = os.path.join("models", "ids_model.pkl")
DEFAULT_PCAP = os.path.join("sample_pcaps", "2026-02-28-traffic-analysis-exercise.pcap")

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    """Load model once and cache it for the session."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

def predict_with_model(model, df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    Run prediction and produce dataframe with columns:
      - malicious_probability
      - prediction (0/1)
      - label (customizable later)
    """
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # Try predict_proba
    try:
        probs = model.predict_proba(df)[:, 1]
    except Exception:
        # If unavailable, fallback to model.predict (not probabilistic)
        preds = model.predict(df)
        probs = [None] * len(preds)
        # Convert to float probabilities if None later
    preds = [1 if (p is not None and p >= threshold) else 0 for p in probs]
    result_df = df.copy()
    result_df["malicious_probability"] = [float(p) if p is not None else None for p in probs]
    result_df["prediction_int"] = preds
    return result_df

def df_to_download_bytes(df: pd.DataFrame, fmt: str = "csv"):
    """Return bytes for download button (csv or json)."""
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    else:
        return df.to_json(orient="records").encode("utf-8")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("Adjust labels, threshold, theme and other options.")

    # Labels
    benign_label = st.text_input("Label for benign traffic", value="BENIGN")
    malicious_label = st.text_input("Label for malicious traffic", value="MALICIOUS")

    # Threshold
    threshold = st.slider("Probability threshold for flagging as malicious", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    # Theme selection (affects subtle colors in the app)
    theme = st.selectbox("Theme", options=["Dark (default)", "Light (compact)"])

    show_raw = st.checkbox("Show raw extracted features", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload / sample", value=True)

    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Run Detection Now 🔎"):
        st.session_state["force_run"] = True

    if st.button("Clear cached model"):
        # clearing cached resource requires special handling in some Streamlit versions
        try:
            del st.session_state["loaded_model"]
        except Exception:
            pass
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("App built with ❤️ — tweak settings and explore results.")

# Load model (wrapped to show friendly error)
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(f"Model missing: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------- Main layout ----------
st.title("🛡️ AI-Powered Intrusion Detection System")
st.markdown("Use the controls on the left to customize detection labels, threshold and display options.")

# Two-column top: Upload + summary
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("Upload PCAP or use sample")
    uploaded_file = st.file_uploader("Upload a PCAP file (.pcap)", type=["pcap"], help="Drag-and-drop or click to select a PCAP.")
    sample_choice = st.selectbox("Or choose a sample PCAP", options=["Use uploaded file", "Default sample"], index=1)

    if uploaded_file is not None:
        # Save uploaded to temp.pcap
        temp_path = "temp.pcap"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        pcap_path = temp_path
    else:
        pcap_path = DEFAULT_PCAP

    # Run automatically if requested
    do_run = st.session_state.get("force_run", False) or run_on_upload or st.button("Extract & Predict ▶️")
    # Clear the forced flag after reading
    st.session_state["force_run"] = False

    # Option: inspect file metadata
    st.markdown("**PCAP file:**")
    st.write(pcap_path)

with colB:
    st.subheader("Status & Quick Info")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"- Model path: `{MODEL_PATH}`")
    st.markdown(f"- Threshold: **{threshold:.2f}**")
    st.markdown(f"- Labels: **{benign_label}** / **{malicious_label}**")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Feature extraction + prediction ----------
if do_run:
    with st.spinner("Extracting features from PCAP (this may take a few seconds)... ⛏️"):
        try:
            df = extract_features_from_pcap(pcap_path)
            # basic sanity
            if df is None or df.shape[0] == 0:
                st.warning("No flows/packets were parsed from the PCAP. Check the PCAP content.")
        except Exception as e:
            st.exception(f"Feature extraction failed: {e}")
            st.stop()

    # Ensure expected features (pad if needed) — keep original behavior
    expected_features = 42
    current_features = df.shape[1]
    if current_features < expected_features:
        for i in range(expected_features - current_features):
            df[f"dummy_{i}"] = 0

    # Run model prediction
    with st.spinner("Running model inference... 🤖"):
        try:
            results_df = predict_with_model(model, df, threshold=threshold)
        except Exception as e:
            st.exception(f"Model prediction failed: {e}")
            st.stop()

    # Map labels
    results_df["label"] = results_df["prediction_int"].apply(lambda x: malicious_label if x == 1 else benign_label)

    # Metrics
    total = len(results_df)
    attacks = int(results_df["prediction_int"].sum())
    normals = total - attacks
    attack_ratio = attacks / total if total > 0 else 0.0

    # Threat level logic (customizable)
    if attack_ratio < 0.1:
        threat = "LOW"
        color = "🟢"
    elif attack_ratio < 0.3:
        threat = "MEDIUM"
        color = "🟠"
    else:
        threat = "HIGH"
        color = "🔴"

    # Top summary cards
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Flows/Rows", total, "📦")
    s2.metric("Benign", normals, "✅")
    s3.metric("Suspicious", attacks, "⚠️")
    s4.metric("Threat Level", f"{threat} {color}", delta=f"{attack_ratio*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts, distribution and probability visualization
    st.subheader("Traffic Distribution")
    dist_df = pd.DataFrame({"label": [benign_label, malicious_label], "count": [normals, attacks]})
    st.bar_chart(dist_df.set_index("label"))

    st.subheader("Attack Probability — sample view")
    prob_df = results_df[["malicious_probability"]].reset_index(drop=True)
    st.line_chart(prob_df)

    # Show the dataframe (with optional highlighting)
    st.subheader("Detected Flows (sample view)")
    if show_raw:
        # Add a small highlight column if requested
        if highlight_suspicious:
            # Add color based on label (pandas style -> streamlit accepts styler)
            def highlight_row(row):
                return ["background-color: rgba(255,0,0,0.12)" if row["prediction_int"] == 1 else "" for _ in row]
            try:
                st.dataframe(results_df.style.apply(highlight_row, axis=1), height=320)
            except Exception:
                st.dataframe(results_df, height=320)
        else:
            st.dataframe(results_df, height=320)
    else:
        # show key columns only
        st.dataframe(results_df[["label", "malicious_probability", "prediction_int"]].rename(columns={
            "malicious_probability":"probability", "prediction_int":"pred_int"
        }), height=320)

    # Row detail explorer
    st.subheader("Inspect a single flow / row")
    idx = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, total-1), value=0, step=1)
    if total > 0:
        row = results_df.iloc[int(idx)].to_dict()
        st.json(row)

    # Downloads and export
    st.subheader("Export results")
    csv_bytes = df_to_download_bytes(results_df, fmt="csv")
    json_bytes = df_to_download_bytes(results_df, fmt="json")
    st.download_button("📥 Download CSV", csv_bytes, file_name="ai_ids_results.csv", mime="text/csv")
    st.download_button("📥 Download JSON", json_bytes, file_name="ai_ids_results.json", mime="application/json")

    # Option to save predictions to models/predictions.csv locally
    if st.button("💾 Save results to predictions.csv"):
        out_path = "predictions.csv"
        results_df.to_csv(out_path, index=False)
        st.success(f"Saved to {out_path}")

    st.success("Detection completed ✅")

# show welcome + demo controls
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Welcome 👋")
    st.markdown(
        "This dashboard lets you upload PCAPs, extract features, and run an ML model to detect suspicious traffic. "
        "Use the sidebar to customize labels and the detection threshold. Click **Run Detection Now** in the sidebar to start."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ by AI-IDS team — tweak the dashboard and explore the results.</div>", unsafe_allow_html=True)