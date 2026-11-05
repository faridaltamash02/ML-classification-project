
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# STEP 1: Load the dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# STEP 2: Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"]

# STEP 3: Reproduce the same 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 4: Identify numerical and categorical features
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

# STEP 7: Combine the preprocessing pipelines
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# STEP 8: Create the Decision Tree pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    ))
])

# STEP 9: Train
model.fit(X_train, y_train)

# STEP 10: Predict
predictions = model.predict(X_test)

# STEP 11: Evaluate
print("Decision Tree Test Accuracy:")
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

# STEP 12: Save the complete fitted pipeline
Path("model").mkdir(exist_ok=True)

joblib.dump(
    model,
    "model/decision_tree.joblib"
)

print("\nDecision Tree model saved successfully!")




tree = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=tree.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classification - First 3 Levels")
plt.tight_layout()
plt.show()
