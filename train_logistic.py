
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"]

# Reproduce the Day 2 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identify feature types
num_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Create complete model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

print("First 10 predictions:")
print(predictions[:10])

print("\nFirst 10 actual labels:")
print(y_test.iloc[:10].values)

# Save the entire trained pipeline
Path("model").mkdir(exist_ok=True)

joblib.dump(
    model,
    "model/logistic_regression.joblib"
)

print("\nLogistic Regression model saved successfully!")

# Evaluate all test observations
accuracy = accuracy_score(y_test, predictions)

print("\nOverall Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["no", "yes"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["no", "yes"],
        zero_division=0
    )
)
