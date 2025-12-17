import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional

CLEANED_PATH = "data/cleaned_telco_churn.csv"
MODEL_PATH = "models/final_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load and return saved pipeline (preprocessor + model).
    Raises FileNotFoundError if path missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MOdel not found at {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline

def get_expected_columns_from_pipeline(pipeline) -> Optional[list]:
    """
    Try to infer the original raw input column names that the preprocessor expects.
    Returns a list of column names or None if it can't infer.
    """
    # pipeline is expected to be a sklearn Pipeline with a 'preprocessor' step
    try:
        pre = pipeline.named_steps['preprocessor', None]
        if pre is None:
            # Fallback to first step if named step not found
            pre = pipeline.steps[0][1]
    except Exception:
        pre = None
    
    if pre is None:
        return None
    
    cols = []
    # ColumnTransformer stores the original 'transformers' attribute as (name, transformer, columns)
    try:
        for name, transformer, transformer_cols in pre.transformers:
            # transformer_cols can be a list/tuple or a slice or string; handle common cases
            if isinstance(transformer_cols, (list, tuple)):
                cols.extend(list(transformer_cols))
            elif isinstance(transformer_cols, str):
                cols.append(transformer_cols)
            # ignore ('remainder', ...) entries or passthrough slices
    except Exception:
        return None

    # remove duplicates while preserving order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered if ordered else None

def prepare_input(input_dict: dict, pipeline=None) -> pd.DataFrame:
    """
    Convert single-customer input dict to a DataFrame suitable for the saved pipeline.
    - Get expected columns from pipeline or from cleaned CSV.
    - For categorical columns fill missing with string 'Missing' so OneHotEncoder sees strings (not float np.nan).
    - For numeric columns coerce to numeric type.
    """
    df = pd.DataFrame([input_dict])

    expected_cols = None
    if pipeline is not None:
        expected_cols = get_expected_columns_from_pipeline(pipeline)

    if expected_cols is None:
        # fallback: read cleaned CSV to infer feature columns (exclude target 'Churn')
        if os.path.exists(CLEANED_PATH):
            tmp = pd.read_csv(CLEANED_PATH, nrows=1)
            expected_cols = [c for c in tmp.columns if c != 'Churn']
        else:
            expected_cols = list(df.columns)

    # ensure all expected columns exist in df, if not create them with NaN
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # If we have the cleaned CSV, inspect dtypes to decide which are categorical
    cat_cols = set()
    num_cols = set()
    if os.path.exists(CLEANED_PATH):
        sample = pd.read_csv(CLEANED_PATH, nrows=100)  # small sample to infer dtypes
        for col in expected_cols:
            if col not in sample.columns:
                continue
            if pd.api.types.is_numeric_dtype(sample[col]):
                num_cols.add(col)
            else:
                cat_cols.add(col)

    # For categorical columns: ensure dtype object and fill missing with a string placeholder
    for c in cat_cols:
        # convert to object dtype and fill missing with a placeholder string
        df[c] = df[c].astype(object)
        df[c] = df[c].where(df[c].notna(), 'Missing')

    # For numeric columns: coerce to numeric (invalid -> NaN)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final ordering: keep only expected_cols in the same order
    df = df[expected_cols]

    return df

def predict_single(input_dict: dict, model_path: str = MODEL_PATH) -> dict:
    """
    1) Load pipeline
    2) Prepare input DataFrame (ensures correct columns)
    3) Call pipeline.predict() and pipeline.predict_proba() if available
    Returns a dict: {'prediction': int(0/1), 'probability': float or None}
    """
    pipeline = load_model(model_path)
    X = prepare_input(input_dict, pipeline=pipeline)

    print("Prepared input dtypes:\n", X.dtypes)
    print("Prepared input values:\n", X.head(1).to_dict(orient='records')[0])

    # predicted label (0/1)
    pred = pipeline.predict(X)
    pred_val = int(pred.ravel()[0])

    # probability for positive class
    prob = None
    try:
        proba = pipeline.predict_proba(X)
        # handle both shape (n,2) and shape (n,1)
        if proba.shape[1] == 1:
            prob = float(proba.ravel()[0])
        else:
            prob = float(proba[:, 1].ravel()[0])
    except Exception:
        # fallback: try decision_function -> sigmoid
        try:
            df_scores = pipeline.decision_function(X)
            # df_scores can be scalar or array-like
            val = np.asarray(df_scores).ravel()[0]
            prob = float(1 / (1 + np.exp(-val)))
        except Exception:
            prob = None

    return {"prediction": pred_val, "probability": prob}


if __name__ == "__main__":
    # Minimal example usage. Update sample to include ALL features your model expects.
    # If you omit some features, they will be filled with NaN (preprocessor should handle/raise).
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.5
        # add other columns your training used if needed
    }

    result = predict_single(sample_input, model_path=MODEL_PATH)
    print("Prediction (0=no churn, 1=churn):", result["prediction"])
    print("Probability of churn (if available):", result["probability"])
        