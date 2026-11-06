
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Import existing data and preprocessing
from process import (
    X_train,
    X_test,
    y_train,
    y_test,
    build_preprocessor
)

# Create GaussianNB pipeline
model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", GaussianNB())
])

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("\nGaussianNB Test Accuracy:")
print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

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

# Save model
Path("model").mkdir(exist_ok=True)

joblib.dump(
    model,
    "model/naive_bayes.joblib"
)

print("\nGaussianNB model saved successfully!")

# Inspect learned probabilities
nb = model.named_steps["classifier"]

print("\nClasses:", nb.classes_)
print("Class priors:", nb.class_prior_)

print("\nFirst five probabilities:")
print(model.predict_proba(X_test.head()))
