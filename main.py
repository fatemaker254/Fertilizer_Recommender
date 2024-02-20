import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("Fertilizer_Prediction.csv")

# Data preprocessing
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

data["Soil_Num"] = data["Soil Type"].map(soil_dict)
data["Crop_Num"] = data["Crop Type"].map(crop_dict)
data = data.drop(["Soil Type", "Crop Type"], axis=1)

# Split the dataset into features and target variable
X = data.drop(["Fertilizer Name"], axis=1)
y = data["Fertilizer Name"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model building
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Model evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Serialize the model using pickle
with open("Fertclassifier.pkl", "wb") as f:
    pickle.dump(classifier, f)


def recommendation(
    temperature,
    humidity,
    moisture,
    nitrogen,
    potassium,
    phosphorous,
    soil_type,
    crop_type,
):
    soil_num = soil_dict[soil_type]
    crop_num = crop_dict[crop_type]

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
    with open("Fertclassifier.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(features).reshape(1, -1)

    return prediction[0]


# Example usage
temperature = 26
humidity = 52
moisture = 38
nitrogen = 37
potassium = 0
phosphorous = 0
soil_type = "Sandy"
crop_type = "Maize"

recommendation = recommendation(
    temperature,
    humidity,
    moisture,
    nitrogen,
    potassium,
    phosphorous,
    soil_type,
    crop_type,
)
print(f"Fertilizer Recommendation: {recommendation}")
