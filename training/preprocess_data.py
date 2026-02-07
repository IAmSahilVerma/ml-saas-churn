import pandas as pd
from training.features import Preprocessor
from training.load_data import load_raw_data

if __name__=="__main__":
    df = load_raw_data()
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(df)
    
    X_processed["Churn"] = df["Churn"]
    
    preprocessor.save("models/preprocessor.pkl")
    X_processed.to_csv("data/processed/churn_processed.csv", index=False)
    
    print("Preprocessing done. Processed shape:", X_processed.shape)