import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

class Preprocessor:
    def __init__(self):
        # Will store transformers
        self.ohe = None
        self.scaler = None
        self.cat_features = [
            "gender", "Partner", "Dependents", "PhoneService", 
            "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        self.num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        
    def fit(self, df: pd.DataFrame):
        # Fill missing numerics
        df[self.num_features] = df[self.num_features].fillna(0)
        # Fit OneHotEncoder
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        df_cat = df[self.cat_features]
        self.ohe.fit(df_cat)
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_features])
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df[self.num_features].fillna(0)
        df_num_scaled = pd.DataFrame(self.scaler.transform(df_num), columns=self.num_features)
        
        df_cat = df[self.cat_features]
        df_cat_encoded = pd.DataFrame(
            self.ohe.transform(df_cat),
            columns=self.ohe.get_feature_names_out(self.cat_features)
        )
        
        df_processed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    def save(self, path="models/preprocessor.pkl"):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path="models/preprocessor.pkl"):
        return joblib.load(path)