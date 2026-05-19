from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# LOAD MODEL
model = joblib.load("football_model.pkl")
#scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    HTGS = float(request.form["HTGS"])
    ATGS = float(request.form["ATGS"])
    HTGD = float(request.form["HTGD"])
    ATGD = float(request.form["ATGD"])
    DiffPts = float(request.form["DiffPts"])

    # INPUT FEATURES
    features = np.array([[HTGS, ATGS, HTGD, ATGD, DiffPts]])
    #features_scaled = scaler.transform(features)
    # PREDICT
    prediction = model.predict(features)[0]

    # RESULT
    if prediction == 'H':
        result = "🏠 Home Team Wins"
    else:
        result = "✈️ Away Team Wins or Draw"

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)