from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_engineering import build_preprocessing_pipeline, split_data
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd
import os
import joblib
def train_models():
    # Load cleaned data (use the cleaned CSV you produced)
    df = pd.read_csv(r"C:\Users\ranji\OneDrive\Desktop\ml_churn_project\data\cleaned_telco_churn.csv")

    # Identify feature lists
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df.columns if col not in numerical_cols + ['Churn']]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)

    # Split data
    x_train, x_test, y_train, y_test = split_data(df)

    # Make sure target is numeric 0/1 (if it is 'Yes'/'No')
    # Simple and safe: map if dtype is object
    if y_train.dtype == 'object':
        y_train = y_train.map({'Yes': 1, 'No': 0})
    if y_test.dtype == 'object':
        y_test = y_test.map({'Yes': 1, 'No': 0})

    # Create model pipelines (preprocessor + model)
    models = {
         'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
         'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
         'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_estimators=100
            )
    }
    
    results = {}
    fitted_pipelines = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        # Fit on training data
        pipe.fit(x_train, y_train)
        # store fitted pipeline
        fitted_pipelines[name] = pipe
        # Predict on test data
        y_pred = pipe.predict(x_test)

        # Evaluate and compare models
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

    # print a simple comparison (done after all models are evaluated)
    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1']:.4f}")
        print()

    # 8. Automatically pick best model (highest F1-score)
    best_name = max(results, key=lambda x: results[x]['f1'])
    print(f"\n Best model selected: {best_name}")

    # 9. Save best model pipeline
    joblib.dump(fitted_pipelines[best_name], "models/final_model.pkl")
    print("Saved best model pipeline to models/final_model.pkl")

    os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    train_models()