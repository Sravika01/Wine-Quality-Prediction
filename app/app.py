from flask import Flask, render_template, request
import joblib
import numpy as np


# Create Flask application
app = Flask(__name__)

# Load trained model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "wine_quality_model.pkl")

model = joblib.load(MODEL_PATH)
#Create the Home Route
@app.route("/")
def home():
    return render_template("index.html")
#Create the Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    features = [

    float(request.form["fixed_acidity"]),
    float(request.form["volatile_acidity"]),
    float(request.form["citric_acid"]),
    float(request.form["residual_sugar"]),
    float(request.form["chlorides"]),
    float(request.form["free_sulfur_dioxide"]),
    float(request.form["total_sulfur_dioxide"]),
    float(request.form["density"]),
    float(request.form["ph"]),
    float(request.form["sulphates"]),
    float(request.form["alcohol"])

]
    prediction = model.predict([features])
    return render_template(
    "index.html",
    prediction_text=f"Predicted Wine Quality: {prediction[0]}"
)

if __name__ == "__main__":
    app.run(debug=True)