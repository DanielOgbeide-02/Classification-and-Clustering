import os
import joblib

# Define the path to your saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory path
MODEL_PATH = os.path.join(BASE_DIR, "src", "webapp", "breast_cancer_model (1).pkl")

def load_model():
    """Load the trained model from MODEL_PATH (raise if missing)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH!r}.")
    return joblib.load(MODEL_PATH)
