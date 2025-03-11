from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Define the input structure using Pydantic (only the relevant features)
class PlayerData(BaseModel):
    appearance: int
    minutes_played: int
    highest_value: float

# Load the KMeans model
modelKMeans = joblib.load('Models/model_KMeans.joblib')  # Corrected model path

# Load the scaler
scaler_classification = joblib.load('Models/scaler_unsupervised.joblib')  # Corrected scaler path

# Define a mapping for the cluster labels to human-readable descriptions
cluster_mapping = {
    0: "Local Football Players",
    1: "Beginner Football Players",
    2: "Professional League Football Players (International)"
}

# Define the prediction endpoint
@app.post("/predict/")
def predict(player_data: PlayerData):
    try:
        # Convert the input data into a numpy array for the model (only using appearance, minutes_played, and highest_value)
        input_data = np.array([
            player_data.appearance,
            player_data.minutes_played,
            player_data.highest_value
        ]).reshape(1, -1)  # Reshape to a 2D array for the model

        # Scale the input data
        scaled_input = scaler_classification.transform(input_data)

        # Make a prediction using KMeans (predicting the cluster)
        cluster_prediction = modelKMeans.predict(scaled_input)

        # Map the cluster number to the human-readable label
        cluster_label = cluster_mapping.get(int(cluster_prediction[0]), "Unknown")

        # Return the predicted cluster and its human-readable label
        return {"predicted_cluster": int(cluster_prediction[0]), "label": cluster_label}

    except Exception as e:
        # Log the error and return a 500 Internal Server Error response
        print(f"Error: {e}")
        return {"error": "Internal Server Error", "details": str(e)}

# Optionally, add a basic endpoint for testing the server
@app.get("/")
def read_root():
    return {"message": "Welcome to the KMeans model prediction API"}
