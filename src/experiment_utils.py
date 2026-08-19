from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import load_data, clean_data

data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def get_split(df=None):
    if df is None:
        df = clean_data(load_data(data_path))
    x = df.drop(columns=["Churn"])
    y = df["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test

def get_three_way_split(df=None):
    if df is None:
        df = clean_data(load_data(data_path))
    x = df.drop(columns=["Churn"])
    y = df["Churn"]
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    return x_train, x_val, x_test, y_train, y_val, y_test

def build_preprocessor(x):
    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = x.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(transformers=[("num", StandardScaler(), numeric_cols),("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),])

def evalute (name, model, x_test, y_test, threshold = 0.5):
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {"experiment": name,"threshold": threshold,"precision": round(precision_score(y_test, y_pred), 3),"recall": round(recall_score(y_test, y_pred), 3),"f1": round(f1_score(y_test, y_pred), 3),"auc": round(roc_auc_score(y_test, y_proba), 3),}