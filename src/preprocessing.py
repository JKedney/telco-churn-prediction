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


if __name__ == "__main__":
    raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cleaned = clean_data(raw)
    print("Shape:", cleaned.shape)
    print("Churn values:", cleaned["Churn"].unique())