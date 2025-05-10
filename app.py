from flask import Flask, request, render_template
from models import load_model
import numpy as np

app = Flask(__name__)
model = load_model()
feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

@app.route("/", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        feats = [float(request.form[name]) for name in feature_names]
        pred = model.predict([feats])[0]
        prob = model.predict_proba([feats])[0][pred]
        label = "Malignant" if pred == 0 else "Benign"
        return render_template("result.html", prediction=label, probability=prob)
    return render_template("form.html", features=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
