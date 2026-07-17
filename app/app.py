from flask import Flask, render_template, request
import joblib
import numpy as np
import threading
import webbrowser

# Create Flask application
app = Flask(__name__)

# Load trained model
model = joblib.load("../models/wine_quality_model.pkl")
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
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")
if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)