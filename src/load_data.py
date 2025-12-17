import pandas as pd

def load_data(path):
    df  = pd.read_csv(path)
    return df

if __name__ == "__main__":
    data_path = r"C:\Users\ranji\OneDrive\Desktop\ml_churn_project\data\telco_churn.csv"
    df = load_data(data_path)

    # Basic checks:
    print(df.head())
    print(df.shape)

    