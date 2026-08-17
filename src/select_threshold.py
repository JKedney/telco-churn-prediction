import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from experiment_utils import get_three_way_split, build_preprocessor, evalute

x_train, x_val, x_test, y_train, y_val, y_test = get_three_way_split()

model = Pipeline(steps=[("preprocessor", build_preprocessor(x_train)),("classifier", LogisticRegression(max_iter=1000)),])
model.fit(x_train, y_train)

candidates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
val_results = [evalute(f"validation @ {t}", model, x_val, y_val, threshold=t) for t in candidates]
val_table = pd.DataFrame(val_results)
print("--- Threshold selection (validation set) ---")
print(val_table.to_string(index=False))

best_row = val_table.loc[val_table["f1"].idxmax()]
best_threshold = best_row["threshold"]
print(f"\nSelected threshold: {best_threshold} (best validation F1 = {best_row['f1']})")


final = evalute("FINAL (test set)", model, x_test, y_test, threshold=best_threshold)
print("\n--- Final performance (test set, evaluated once) ---")
print(pd.DataFrame([final]).to_string(index=False))