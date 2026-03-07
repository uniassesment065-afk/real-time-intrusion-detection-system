"""
Real-time Intrusion Prediction Module
-------------------------------------
This module loads a trained ML model for intrusion detection
and predicts whether network traffic is benign or malicious.
Labels can be customized dynamically.
"""

import joblib
import pandas as pd

def load_model(model_path="models/ids_model.pkl"):
    """
    Load the trained intrusion detection model from disk.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        model: Loaded ML model object.
    """
    model = joblib.load(model_path)
    return model


def predict_traffic(
    sample_data, 
    model_path="models/ids_model.pkl", 
    benign_label="BENIGN", 
    malicious_label="MALICIOUS"
):
    """
    Predict intrusion on the given network traffic data with customizable labels.
    
    Args:
        sample_data (pd.DataFrame or array-like): Input features for prediction.
        model_path (str): Path to the saved model file.
        benign_label (str): Label for normal/benign traffic.
        malicious_label (str): Label for malicious traffic.
    
    Returns:
        List[dict]: Each dict contains 'prediction' (custom label) and
                    'malicious_probability' (float between 0-1).
    """
    model = load_model(model_path)


    # Ensure input is a DataFrame
    if not isinstance(sample_data, pd.DataFrame):
        sample_data = pd.DataFrame(sample_data)

    predictions = model.predict(sample_data)
    
    # Handle models that may not support predict_proba
    try:
        probabilities = model.predict_proba(sample_data)[:, 1]
    except AttributeError:
        probabilities = [None] * len(predictions)

    results = []
    for i in range(len(predictions)):
        label = malicious_label if predictions[i] == 1 else benign_label
        results.append({
            "prediction": label,
            "malicious_probability": float(probabilities[i]) if probabilities[i] is not None else None
        })

    return results