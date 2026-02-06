import pandas as pd

def load_raw_data(path="data/raw/telco_churn.csv"):
    """
    Load raw churn dataset
    """
    df = pd.read_csv(path)
    return df

if __name__=="__main__":
    df = load_raw_data()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())