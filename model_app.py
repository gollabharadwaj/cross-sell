from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Input model
class Input(BaseModel):
    Gender: str
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    Response: int

# Define Output model
class Output(BaseModel):
    is_promoted: int

# Load your model
try:
    model = joblib.load('promote_pipeline_model.pkl')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Internal Server Error - Model loading failed.")

@app.post("/predict", response_model=Output)
def predict(data: Input):
    try:
        # Prepare input data for prediction
        X_input = pd.DataFrame([[
            data.Gender, data.Age, data.Driving_License, data.Region_Code,
            data.Previously_Insured, data.Vehicle_Age, data.Vehicle_Damage,
            data.Annual_Premium, data.Policy_Sales_Channel, data.Vintage, data.Response
        ]], columns=[
            'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
            'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 
            'Vintage', 'Response'
        ])
        logger.info(f"Input data: {X_input}")
        
        # Model prediction
        prediction = model.predict(X_input)
        logger.info(f"Prediction result: {prediction}")
        
        return Output(is_promoted=int(prediction[0]))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error - Prediction failed.")
