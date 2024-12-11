from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# Define the input data schema
class FlatCharacteristics(BaseModel):
    Property_type: str
    postal_code: int
    size: float
    floor: int
    land_size: float
    energy_performance_category: str
    exposition: str
    nb_rooms: int
    nb_bedrooms: int
    nb_bathrooms: int
    nb_parking_places: int
    nb_boxes: int
    has_a_balcony: int
    nb_terraces: int
    has_air_conditioning: int

# Load the trained model
rf_model = joblib.load('models/random_forest_model.pk1')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_price(data: FlatCharacteristics):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])
        #print("Input Data Before Alignment:\n", input_data)

        # Align the input with the model’s expected features
        input_data = input_data.reindex(columns=rf_model.feature_names_in_, fill_value=0)
        #print("Input Data After Alignment:\n", input_data)

        # Make a prediction
        predictions = rf_model.predict(input_data)
        #print("Predictions Array:", predictions)

        # Extract the first prediction
        predicted_price = predictions[0]  # Same logic as the standalone script
        print("Predicted Price:", predicted_price)

        # Return the prediction
        return {"predicted_price": float(predicted_price)}  # Ensure JSON-serializable
    except Exception as e:
        print("Prediction Error:", str(e))
        return {"error": f"Prediction failed: {str(e)}"}