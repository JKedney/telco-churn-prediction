import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from preprocessing import load_data, clean_data, engineer_features
import os
import joblib
from experiment_utils import *

x_train, x_test, y_train, y_test = get_split()
preprocessor = build_preprocessor(x_train)

def make_pipline(classifer):
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifer)])

def baseline():
    model = make_pipline(LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    return evalute("Logistic Regression (baseline)", model, x_test, y_test, 0.5), model

def balanced():
    model= make_pipline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(x_train, y_train)
    return evalute("logreg + class weight", model, x_test, y_test), model

def grediant_boosting():
    model = make_pipline(HistGradientBoostingClassifier(random_state=42))
    model.fit(x_train, y_train)
    return evalute("Gradient Boosting", model, x_test, y_test), model

def tuned_logreg():
    grid = {"classifier__C": [0.01, 0.1, 1, 10], "classifier__class_weight": [None, "balanced"]}
    search = GridSearchCV(make_pipline(LogisticRegression(max_iter=1000)), grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="roc_auc")
    search.fit(x_train, y_train)
    best = search.best_params_
    name = f"Tuned LogReg (C={best['classifier__C']}, cw={best['classifier__class_weight']})"
    return evalute(name, search.best_estimator_, x_test, y_test), search.best_estimator_

def feature_engineering():
    df = engineer_features(clean_data(load_data(data_path)))
    xtr,xte,ytr,yte = get_split(df)
    pre = build_preprocessor(xtr)
    model = Pipeline(steps=[("preprocessor", pre), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced" ))])
    model.fit(xtr, ytr)
    return evalute("Feature Engineering logreg", model, xte, yte), model

if __name__ == "__main__":
    experiments = [baseline, balanced, grediant_boosting, tuned_logreg, feature_engineering]
    results = []
    models = {}
    for exp in experiments:
        metrics, model = exp()
        results.append(metrics)
        models[metrics["experiment"]] = model

    base_model = models["Logistic Regression (baseline)"]
    for t in [0.4, 0.3]:
        results.append(
            evalute(f"Baseline @ threshold {t}", base_model, x_test, y_test, threshold=t)
        )

    table = pd.DataFrame(results)
    print(table.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/comparison.csv", index=False)
    print("\nSaved to results/comparison.csv")
    os.makedirs("models", exist_ok=True)
    joblib.dump(base_model, "models/churn_model.joblib")
    print("Saved model to models/churn_model.joblib (use at threshold 0.3)")
