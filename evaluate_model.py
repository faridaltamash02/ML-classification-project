
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ----------------------------------------
# STEP 1: Define saved model locations
# ----------------------------------------
model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


# ----------------------------------------
# STEP 2: Load the original test data
# ----------------------------------------
test_df = pd.read_csv("data/test_data.csv")

if "y" not in test_df.columns:
    raise ValueError("Test CSV must contain target column 'y'.")

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"]

if not set(y_test.unique()).issubset({"no", "yes"}):
    raise ValueError("Target must contain only 'no' or 'yes'.")

if y_test.nunique() != 2:
    raise ValueError("ROC AUC requires both classes in the test data.")

y_test_binary = (y_test == "yes").astype(int)

print("Test dataset:", test_df.shape)


# ----------------------------------------
# STEP 3: Evaluate each saved model
# ----------------------------------------
results = []

for model_name, model_path in model_paths.items():

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Missing saved model: {model_path}"
        )

    model = joblib.load(model_path)

    # Predictions: no / yes
    predictions = model.predict(X_test)

    # Continuous scores for AUC
    if hasattr(model, "predict_proba"):

        class_order = list(model.classes_)
        yes_index = class_order.index("yes")

        scores = model.predict_proba(X_test)[:, yes_index]

    elif hasattr(model, "decision_function"):

        class_order = list(model.classes_)

        if len(class_order) != 2 or "yes" not in class_order:
            raise ValueError(
                f"Unexpected classes for {model_name}: "
                f"{class_order}"
            )

        scores = model.decision_function(X_test)

        # Binary decision_function is oriented
        # toward classes_[1].
        if class_order[1] != "yes":
            scores = -scores

    else:
        raise ValueError(
            f"{model_name} cannot provide AUC scores."
        )

    # Six required metrics
    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(y_test_binary, scores)

    precision = precision_score(
        y_test, predictions, pos_label="yes",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions, pos_label="yes",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, pos_label="yes",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, predictions)

    # Store results
    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Print confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["no", "yes"]
    )

    print(f"\n{model_name}")
    print("Confusion Matrix:")
    print(cm)


# ----------------------------------------
# STEP 4: Create comparison table
# ----------------------------------------
comparison_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(
    comparison_df.round(4).to_string(index=False)
)


# ----------------------------------------
# STEP 5: Save comparison table
# ----------------------------------------
Path("results").mkdir(exist_ok=True)

comparison_df.round(6).to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("results/model_comparison.csv")
