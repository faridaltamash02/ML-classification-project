# Bank Marketing — ML Classification Model Comparison

A machine learning classification assignment using the **UCI Bank Marketing** dataset to compare five classifiers and deploy an interactive Streamlit application.

> **Project status:** Day 1 — dataset selection and exploratory data analysis (EDA) completed. Preprocessing, model training, evaluation, and Streamlit deployment are planned and **not yet completed**.

## Project objective

Predict whether a bank customer will subscribe to a term deposit (`y = yes` or `y = no`) using customer and marketing campaign information. Build and compare the following classifiers:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Evaluate them on a **common held-out test set** using Accuracy, ROC AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## Dataset

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **File:** `data/bank-full.csv` (semicolon-separated)
- **Observations:** 45,211
- **Columns:** 17 (16 input features and one target)
- **Target:** `y` — `yes` (subscribed) or `no` (not subscribed)
- **Initial numeric features (7):** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- **Initial categorical features (9):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

**Prediction-time constraint:** `duration` (length of the last contact) will be excluded from the model if predicting subscription **before** the call. This leaves 15 candidate input features, exceeding the assignment minimum of 12.

## Day 1 — EDA findings

### Target class distribution

| Target | Count | Percentage |
|---|---:|---:|
| `no` | 39,922 | 88.30% |
| `yes` | 5,289 | 11.70% |

The dataset is **class imbalanced**. Accuracy alone can be misleading: an illustrative baseline that always predicts `no` would obtain approximately **88.30% accuracy** while identifying **none** of the `yes` observations.

### Missing and `unknown` values

`df.isnull().sum()` returned **0 for every column**. However, some fields contain an explicit `unknown` category:

| Column | `unknown` count | Share of all rows |
|---|---:|---:|
| `job` | 288 | 0.64% |
| `education` | 1,857 | 4.11% |
| `contact` | 13,020 | 28.80% |
| `poutcome` | 36,959 | 81.75% |

These are strings, not pandas null values. Initially, the preprocessing plan is to preserve `unknown` as a category and reconsider treatment after evaluating its meaning and predictive utility.

### Numerical feature observations

- **Age:** mean ≈ 40.94 years, median 39; most records involve middle-aged customers.
- **Balance:** mean ≈ 1,362.27 and median 448; highly right-skewed, with values ranging from −8,019 to 102,127.
- **Duration:** mean ≈ 258.16 seconds, median 180; right-skewed. Exclude for pre-call prediction to avoid using unavailable future information.
- **Campaign:** median 2 and maximum 63; a long right tail.
- **Previous:** median 0 and maximum 275; concentrated at zero.
- **Pdays:** `-1` is a special value indicating no previous contact, not a regular negative elapsed time. It occurs in **36,954** rows; **8,257** rows have other values.

### Education and subscription rate

| Education category | Subscribed (`yes`) rate |
|---|---:|
| Tertiary | 15.01% |
| Unknown | 13.57% |
| Secondary | 10.56% |
| Primary | 8.63% |

The tertiary group has a **6.38 percentage-point** higher subscription rate than the primary group. This is an **association**, not proof of causation. `unknown` is not an education level; it indicates unspecified information.

### EDA visualizations

Generated during Day 1:

- Numerical feature histograms
- Subscription rate by education bar chart

Save charts in `results/` (for example, `numerical_distributions.png` and `education_subscription.png`) to include them in the report.

## Proposed project structure

```text
classification-project/
├── app.py                         # Planned: Streamlit interface
├── explore_data.py                # Day 1: EDA
├── preprocess.py                  # Planned: shared preprocessing
├── train_models.py                # Planned: training and saving models
├── evaluate_models.py             # Planned: evaluation and comparison
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── bank-full.csv
│   └── test_data.csv               # Created on Day 2
├── model/                          # Saved pipelines (Day 3–4)
└── results/
    ├── numerical_distributions.png
    ├── education_subscription.png
    └── model_comparison.csv        # Created on Day 4
```

*Files labeled “Planned” are placeholders for the intended structure; they may not exist yet.*

## Installation and Day 1 usage

Requires a compatible Python 3 environment. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Download and extract the UCI Bank Marketing dataset, then place `bank-full.csv` under `data/`.

```bash
python explore_data.py
```

The input CSV uses semicolons:

```python
import pandas as pd

df = pd.read_csv("data/bank-full.csv", sep=";")
print(df.shape)
print(df["y"].value_counts())
```

Expected shape: `(45211, 17)`.

## Development plan

| Day | Deliverable | Status |
|---|---|---|
| 1 | Dataset selection, EDA, repository organization | EDA completed; repository setup to verify |
| 2 | Train/test split, encoding, scaling, leak-resistant pipelines | Planned |
| 3 | Logistic Regression, Decision Tree, KNN | Planned |
| 4 | Gaussian Naive Bayes, SVM, metrics, confusion matrices | Planned |
| 5 | Streamlit app, CSV upload, model selection and deployment | Planned |
| 6 | BITS Virtual Lab execution, screenshot, final README and submission PDF | Planned |

## Methodology (planned)

1. Separate `X` and `y`, excluding `duration` for the defined pre-call prediction task.
2. Create a reproducible stratified train/test split (`random_state=42`).
3. Build preprocessing with `ColumnTransformer` and `Pipeline`. Fit preprocessing on training data only to prevent leakage.
4. Encode categorical features, handle missing/unknown values appropriately, and scale numerical features when needed.
5. Train all five models and save **complete fitted pipelines** using `joblib`.
6. Evaluate all models on the **same test set**. Calculate Accuracy, ROC AUC, Precision, Recall, F1, MCC and confusion matrices. Explicitly state the positive class (`yes`).
7. Deploy a Streamlit app to load saved models and score uploaded labeled CSV data without retraining.

### Model comparison table (to populate after evaluation)

| Model | Accuracy | ROC AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Gaussian Naive Bayes | TBD | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD | TBD |

**Note:** For the binary target, ROC AUC will use scores/probabilities for the `yes` class. Model selection should consider class imbalance, not just accuracy.

## Streamlit application (planned)

Expected features:

- Upload a labeled CSV file
- Validate required input columns and target values
- Choose one of five saved classifiers
- Display six evaluation metrics and a confusion matrix
- Run inference from saved fitted models without retraining

Deployment URL: **TBD**

## Submission checklist

- [ ] GitHub repository link added
- [ ] Streamlit deployment URL added
- [ ] All five trained model files committed
- [ ] `test_data.csv` included
- [ ] Model comparison table populated with actual results
- [ ] `requirements.txt` tested
- [ ] BITS Virtual Lab execution screenshot captured
- [ ] Final assignment PDF assembled

## Reproducibility notes

Keep the test set separate from training and model-selection activities. Do not use test data to fit scalers, imputers or encoders. Once the project works, pin tested dependency versions before final submission.

## Dataset attribution

Dataset: **Bank Marketing**, UCI Machine Learning Repository. See the [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing) for the original authors, citation guidance, license, and dataset details.
