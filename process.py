
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.base import clone

# -------------------------------------------
# STEP 1: Load the dataset
# -------------------------------------------
df = pd.read_csv(
    "data/bank-full.csv",
    sep=";"
)

print("Original dataset:", df.shape)


# -------------------------------------------
# STEP 2: Separate features and target
# -------------------------------------------
# Exclude duration for pre-call predictions.
X = df.drop(columns=["y", "duration"])
y = df["y"]

print("Features:", X.shape)
print("Target:", y.shape)


# -------------------------------------------
# STEP 3: Train/Test Split
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------------------------
# STEP 4: Identify feature types
# -------------------------------------------
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical:", numerical_features)
print("Categorical:", categorical_features)


# -------------------------------------------
# STEP 5: Numerical pipeline
# -------------------------------------------
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# -------------------------------------------
# STEP 6: Categorical pipeline
# -------------------------------------------
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# -------------------------------------------
# STEP 7: Combine both pipelines
# -------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# -------------------------------------------
# STEP 8: Fit on training data only
# -------------------------------------------
X_train_processed = preprocessor.fit_transform(
    X_train
)

# Transform test data without refitting
X_test_processed = preprocessor.transform(
    X_test
)


# -------------------------------------------
# STEP 9: Validate transformed data
# -------------------------------------------
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print(
    "Processed training:",
    X_train_processed.shape
)

print(
    "Processed testing:",
    X_test_processed.shape
)

print(
    "\nFeature names:",
    preprocessor.get_feature_names_out()
)


# -------------------------------------------
# STEP 10: Save RAW test data
# -------------------------------------------
Path("data").mkdir(exist_ok=True)

test_data = X_test.copy()
test_data["y"] = y_test

test_data.to_csv(
    "data/test_data.csv",
    index=False
)

print("\nTest data saved successfully!")

def build_preprocessor():
    """
    Return a fresh, unfitted copy of the
    preprocessing pipeline.
    """
    return clone(preprocessor)
