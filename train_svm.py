
import joblib
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Reuse our existing Day 2 split and preprocessing
from process import (
    X_train,
    X_test,
    y_train,
    y_test,
    build_preprocessor
)

# STEP 1: Create the complete SVM pipeline
svm_model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        cache_size=500
    ))
])

# STEP 2: Train SVM
svm_model.fit(X_train, y_train)

# STEP 3: Generate class predictions
predictions = svm_model.predict(X_test)

# STEP 4: Generate decision scores for AUC
decision_scores = svm_model.decision_function(X_test)

# STEP 5: Evaluate
accuracy = accuracy_score(y_test, predictions)

auc = roc_auc_score(
    (y_test == "yes").astype(int),
    decision_scores
)

print("\nSVM Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nSVM ROC AUC:")
print(f"{auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["no", "yes"]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["no", "yes"],
    zero_division=0
))

# STEP 6: Save trained pipeline
Path("model").mkdir(exist_ok=True)

joblib.dump(
    svm_model,
    "model/support_vector_machine.joblib"
)

print("\nSVM model saved successfully!")

# STEP 7: Inspect trained classifier
svc = svm_model.named_steps["classifier"]

print("\nClass order:")
print(svc.classes_)

print("\nSupport vectors per class:")
print(svc.n_support_)

print("\nTotal support vectors:")
print(svc.support_vectors_.shape[0])

print("\nFirst five predictions:")
print(predictions[:5])

print("\nFirst five decision scores:")
print(np.round(decision_scores[:5], 4))

svc = svm_model.named_steps["classifier"]

print(svc.n_support_)
