import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"])
    return df

def engineer_features(df):
    df = df.copy()
    df["charges_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    return df

if __name__ == "__main__":
    raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cleaned = clean_data(raw)
    print("Shape:", cleaned.shape)
    print("Churn values:", cleaned["Churn"].unique())