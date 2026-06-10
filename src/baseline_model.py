from preprocessing import load_data, clean_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load and clean the data
df = clean_data(load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))

# Split into features (x) and target (y)
x = df.drop(columns=["Churn"])
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", x_train.shape[0])
print("Test rows:", x_test.shape[0])

# Separate columns into numbers vs categories
numeric_cols = x.select_dtypes(include="number").columns.tolist()
categorical_cols = x.select_dtypes(exclude="number").columns.tolist()

# Scale the numbers, one-hot encode the categories
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# Bundle preprocessing + model into one object
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Train it
model.fit(x_train, y_train)
print("Model trained successfully.\n")

# Evaluate on the unseen test set
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))