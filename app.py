from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        Airline = int(request.form["Airline"])
        Source = int(request.form["Source"])
        Destination = int(request.form["Destination"])
        Total_Stops = int(request.form["Total_Stops"])
        Duration = int(request.form["Duration"])
        Journey_day = int(request.form["Journey_day"])
        Journey_month = int(request.form["Journey_month"])

        features = np.array([[Airline, Source, Destination,
                              Total_Stops, Duration,
                              Journey_day, Journey_month]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        output = round(prediction[0], 2)

        return render_template("index.html",
                               prediction_text=f"Estimated Price: ₹ {output}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text="Something went wrong!")

if __name__ == "__main__":
    app.run(debug=True)