"""
Data cleaning and feature engineering for BigMart sales prediction
"""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and normalise inconsistent categories
    """
    df = df.copy()

    #Impute Item_Weight with the mean per item type
    if "Item_Weight" in df.columns:
        df["Item_Weight"] = df.groupby("Item_Type")["Item_Weight"].transform(
            lambda x: x.fillna(x.mean())
        )
        df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())

    #Impute Outlet_Size with the mode per outlet type
    if "Outlet_Size" in df.columns:
        mode_by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Medium"
        )
        df["Outlet_Size"] = df.apply(
            lambda r: mode_by_type[r["Outlet_Type"]]
            if pd.isna(r["Outlet_Size"]) else r["Outlet_Size"],
            axis=1
        )
    
    # Normalise inconsistent fat-content labels
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
            "low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular",
        })

    # A visibility of 0 is impossible - treat as missing, impute with mean
    if "Item_Visibilty" in df.columns:
        df["Item_Visibility"] = df["Item_Visibility"].replace(0, df["Item_Visibility"].mean())

    return df

def engineer_features(df: pd.DataFrame, reference_year: int = 2013) -> pd.DataFrame:
    """
    Create new features from the raw columns.
    """
    df = df.copy()

    # Outlet age is more meaningful than establishment year
    if "Outlet_Establishment_Year" in df.columns:
        df["Outlet_Age"] = reference_year - df["Outlet_Establishment_Year"]

    # Broad item category from the item identifier prefix (FD/DR/NC)
    if "Item_Identifier" in df.columns:
        df["Item_Category"] = df["Item_Identifier"].str[:2].map({
            "FD": "Food", "DR": "Drinks", "NC": "Non-Consumable",
        })

    return df

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns. Return the frame and the encoders
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    cat_cols = df.select_dtypes(include="object").columns
    # Do not encode the raw identifier
    cat_cols = [c for c in cat_cols if c != "Item_Identifier"]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders

def preprocess(path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> clean -> engineer -> encode.
    """
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_categoricals(df)
    # Drop the raw identifier (not useful numerical feature)
    df = df.drop(columns={"Item_Identifier"}, errors="ignore")
    return df
