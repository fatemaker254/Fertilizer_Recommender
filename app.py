from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open("Fertclassifier.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user inputs from the form
        temperature = int(request.form["temperature"])
        humidity = int(request.form["humidity"])
        moisture = int(request.form["moisture"])
        nitrogen = int(request.form["nitrogen"])
        potassium = int(request.form["potassium"])
        phosphorous = int(request.form["phosphorous"])
        soil_type = request.form["soil_type"]
        crop_type = request.form["crop_type"]

        # Map soil type and crop type to numerical values
        soil_dict = {"Loamy": 1, "Sandy": 2, "Clayey": 3, "Black": 4, "Red": 5}

        crop_dict = {
            "Sugarcane": 1,
            "Cotton": 2,
            "Millets": 3,
            "Paddy": 4,
            "Pulses": 5,
            "Wheat": 6,
            "Tobacco": 7,
            "Barley": 8,
            "Oil seeds": 9,
            "Ground Nuts": 10,
            "Maize": 11,
        }

        soil_num = soil_dict[soil_type]
        crop_num = crop_dict[crop_type]

        # Make prediction using the model
        features = np.array(
            [
                [
                    temperature,
                    humidity,
                    moisture,
                    nitrogen,
                    potassium,
                    phosphorous,
                    soil_num,
                    crop_num,
                ]
            ]
        )
        predictions = model.predict_proba(features)[
            0
        ]  # Get the probability distribution
        top_indices = np.argsort(predictions)[::-1][:5]  # Get top 5 indices
        top_5_fertilizers = [
            model.classes_[i] for i in top_indices
        ]  # Get names of top 5 fertilizers

        return render_template(
            "index.html", prediction=top_5_fertilizers, show_result=True
        )

    return render_template("index.html", show_result=False)


if __name__ == "__main__":
    app.run(debug=True)
