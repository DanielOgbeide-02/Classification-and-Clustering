from flask import Flask, request, render_template
from models import load_model
import numpy as np

app = Flask(__name__)
model = load_model()
feature_names = ['worst area', 'worst concave points', 'mean concave points',
     'worst radius', 'worst perimeter', 'mean perimeter', 'mean concavity',
     'mean area', 'worst concavity', 'mean radius', 'area error', 'worst compactness']


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
