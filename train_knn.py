
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# STEP 2: Separate features and target
# Exclude duration for pre-call prediction
X = df.drop(columns=["y", "duration"])
y = df["y"]

# STEP 3: Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 4: Identify feature types
num_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# STEP 5: Numerical preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# STEP 6: Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# STEP 7: Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# STEP 8: Create KNN model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ))
])

# STEP 9: Train the model
model.fit(X_train, y_train)

# STEP 10: Generate predictions
predictions = model.predict(X_test)

# STEP 11: Evaluate
print("\nKNN Test Accuracy:")
print(f"{accuracy_score(y_test, predictions)*100:.2f}%")

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

# STEP 12: Save complete fitted pipeline
Path("model").mkdir(exist_ok=True)

joblib.dump(
    model,
    "model/knn.joblib"
)

print("\nKNN model saved successfully!")


# Estimate class probabilities
probabilities = model.predict_proba(X_test.head())

# Check the class order
print("Class order:")
print(model.classes_)

print("\nFirst five predicted probabilities:")
print(probabilities)
