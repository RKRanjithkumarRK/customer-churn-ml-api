from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

from src.predict import prepare_input

app = FastAPI(
    title = "customer_churn_prediction_api",
    description="Predict customer churn using trained ML model",
    version="1.0"
)

MODEL_PATH = "models/final_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

pipeline = joblib.load(MODEL_PATH)

class ChurnRequest(BaseModel):
    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: str | None = None
    Dependents: str | None = None
    tenure: int
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")

def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")

def predict(request: ChurnRequest):
    try:
        input_dict = request.model_dump()
        x = prepare_input(input_dict, pipeline=pipeline)
        prediction = int(pipeline.predict(x)[0])

        try:
            proba = pipeline.predict_proba(x)[0]
            probability = float(proba[:,1][0])
        except Exception:
            score = pipeline.decision_function(x)[0]
            probability = float(1/(1+np.exp(-score)))

        return {
            "prediction": prediction,
            "probability": probability
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))