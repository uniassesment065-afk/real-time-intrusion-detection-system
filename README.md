# AI_IDPS_PROJECT

## Overview
AI_IDPS_PROJECT is an AI-Powered Intrusion Detection and Prevention System for **offline log analysis**. It parses PCAP files using Zeek, applies a machine-learning model to detect suspicious flows, and presents results in a Streamlit dashboard with explanations.

## Folder Structure
- `data/` : stores datasets (CSV or PCAP).
- `models/` : stores trained ML models.
- `src/` : contains modular Python scripts.
- `notebooks/` : optional Jupyter notebooks for testing and experiments.
- `zeek_logs/` : stores Zeek-generated logs.
- `main.py` : main entry point to run the project.
- `requirements.txt` : Python dependencies.
- `venv/` : virtual environment (ignored by Git).

## Features
- Upload and parse PCAP files offline.
- Feature extraction for ML analysis.
- Detect benign and malicious flows with LightGBM.
- SHAP explainability for alerts.
- Streamlit dashboard to visualize alerts and summaries.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv