from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        overall_qual = float(request.form["overall_qual"])
        gr_liv_area = float(request.form["gr_liv_area"])
        garage_cars = float(request.form["garage_cars"])
        full_bath = float(request.form["full_bath"])
        year_built = float(request.form["year_built"])

        features = np.array([[
            overall_qual,
            gr_liv_area,
            garage_cars,
            full_bath,
            year_built
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        output = round(prediction[0], 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ₹ {output}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)