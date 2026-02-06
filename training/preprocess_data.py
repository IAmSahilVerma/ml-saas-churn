import pandas as pd
from features import Preprocessor
from load_data import load_raw_data

if __name__=="__main__":
    df = load_raw_data()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(df)
    
    preprocessor.save("models/preprocessor.pkl")
    X_processed.to_csv("data/processed/churn_processed.csv", index=False)
    
    print("Preprocessing done. Processed shape:", X_processed.shape)