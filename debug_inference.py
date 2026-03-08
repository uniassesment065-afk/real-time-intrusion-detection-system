import joblib, pandas as pd, numpy as np
from pcap_feature_extractor import extract_features_from_pcap

MODEL_PATH = "models/ids_model.pkl"
TRAIN_CSV = "data/UNSW_NB15_training-set.csv"

# 1) model info
model = joblib.load(MODEL_PATH)
print("Model type:", type(model))
print("Model classes_:", getattr(model, "classes_", None))
if hasattr(model, "feature_names_in_"):
    print("feature_names_in_ length:", len(model.feature_names_in_))
elif hasattr(model, "booster_"):
    try:
        print("LGBM booster.feature_name() sample:", model.booster_.feature_name()[:10])
    except Exception:
        pass

# 2) extract features from the malicious pcap you uploaded
df = extract_features_from_pcap("/home/tobijoshua/projects/AI-IDPS-Project/sample_pcap/2026-02-28-traffic-analysis-exercise.pcap", num_packets=100)
print("Extracted df shape:", df.shape)
print(df.columns.tolist()[:20])
print(df.head(3).T)

# 3) training feature snapshot (if available)
try:
    train = pd.read_csv(TRAIN_CSV)
    cols = [c for c in train.columns if c not in ["id","label","attack_cat"]]
    print("Training columns sample:", cols[:20])
    common = [c for c in df.columns if c in cols]
    print("Common columns count:", len(common))
except Exception as e:
    print("No train CSV available or failed to read:", e)