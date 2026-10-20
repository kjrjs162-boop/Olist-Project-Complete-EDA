from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI app
app = FastAPI(title="E-Commerce ML API", description="API for predicting late deliveries and churn")

# 2. Load the trained model
try:
    rf_model = joblib.load('rf_late_delivery_model.pkl')
except FileNotFoundError:
    rf_model = None

# 3. Define the input data schema
class DeliveryRequest(BaseModel):
    price: float
    freight_value: float
    promised_delivery_days: int

# 4. Create the Prediction Endpoint
@app.post("/predict/late_delivery")
def predict_late_delivery(request: DeliveryRequest):
    if rf_model is None:
        return {"error": "Model not found. Please train and save the model first."}
    
    # Convert incoming JSON request to DataFrame
    input_data = pd.DataFrame([{
        'price': request.price,
        'freight_value': request.freight_value,
        'promised_delivery_days': request.promised_delivery_days
    }])
    
    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]
    
    result = "Late" if prediction == 1 else "On Time"
    
    return {
        "prediction": result,
        "late_probability": f"{probability * 100:.2f}%"
    }