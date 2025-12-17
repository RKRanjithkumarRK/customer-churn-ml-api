import pandas as pd

def clean_data(df, save_clean: bool = False, cleaned_path: str = "data/cleaned_telco_churn.csv"):

    # 1) Convert TotalCharges to numeric (coerce bad values -> NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2) Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 3) Replace service-related strings with "No"
    replace_map = {
        'No internet service': 'No',
        'No phone service': 'No'
    }
    df = df.replace(replace_map)

    # 4) Detect Yes/No columns
    yes_no_cols = [col for col in df.columns if set(df[col].dropna().unique()) <= {'Yes', 'No'}]

    # 5) Convert Yes/No to 1/0
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # 6) Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 7) Save cleaned CSV if needed
    if save_clean:
        df.to_csv(cleaned_path, index=False)

    return df


if __name__ == "__main__":
    # ⭐ FIXED PATHS (your request)
    raw_path = "data/telco_churn.csv"
    cleaned_path = "data/cleaned_telco_churn.csv"

    df = pd.read_csv(raw_path)
    df_clean = clean_data(df, save_clean=True, cleaned_path=cleaned_path)

    print("Saved cleaned CSV to data/cleaned_telco_churn.csv")
    print(df_clean.info())
