import pickle
import pandas as pd
import numpy as np
from pcap_feature_extractor import extract_features_from_pcap

# Load model from dashboard.py session if available
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None

def detect_pcap(pcap_file, num_packets=50):  
    # Extract features from pcap
    df = extract_features_from_pcap(pcap_file, num_packets=num_packets)
    
    if model:
        try:
            preds = model.predict_proba(df)[:, 1]
        except:
            # Shape mismatch or model error
            preds = np.zeros(len(df))
    else:
        preds = np.zeros(len(df))
    
    # Build results list
    results = []
    for p in preds:
        results.append({
            "prediction": "ATTACK" if p > 0.5 else "NORMAL",
            "attack_probability": float(p)
        })
    return results

if __name__ == "__main__":
    import sys
    pcap_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pcap"
    print(detect_pcap(pcap_file, num_packets=50))