# src/webapp/models.py

import os, joblib
from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.pkl")

def train_and_save():
    bc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=0, stratify=bc.target
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH!r}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH!r}.")
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train_and_save()
