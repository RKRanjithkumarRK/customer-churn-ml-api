# src/feature_engineering.py
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Create a ColumnTransformer that scales numerical columns and
    one-hot-encodes categorical columns.
    Returns the preprocessor object.
    """
    # numeric transformer: scale numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # categorical transformer: one-hot encode (ignore unknowns at transform time)
    # sparse=False returns a dense numpy array
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # drop any other columns not specified
    )

    return preprocessor
    
def split_data(df, test_size=0.2, random_state=42):
    """
    Split df into x_train, x_test, y_train, y_test.
    Uses stratify=y so class distribution is preserved.
    Expects df['Churn'] to be numeric (0/1). If not, map before calling this.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # Use cleaned CSV produced by your preprocessing step
    df = pd.read_csv("data/cleaned_telco_churn.csv")

    # Ensure the target is numeric 0/1 (safety)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # define columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df.columns if col not in numerical_cols + ['Churn']]

    # build pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)

    # split data (stratified)
    x_train, x_test, y_train, y_test = split_data(df)

    # fit only on training data
    preprocessor.fit(x_train)

    # transform
    x_train_transformed = preprocessor.transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    print("Train shape:", x_train_transformed.shape)
    print("Test shape:", x_test_transformed.shape)
