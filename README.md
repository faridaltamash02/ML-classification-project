# Bank Marketing — ML Classification Model Comparison

A supervised machine learning project using the **UCI Bank Marketing** dataset to compare five binary classifiers and evaluate them through a deployed Streamlit application.

**Project status:** Model development, evaluation, Streamlit Cloud deployment, and execution in the BITS Virtual Lab have been completed. The final submission PDF is being prepared.

**Project links:**
- **GitHub repository:** https://github.com/faridaltamash02/ml-classification-project
- **Live Streamlit application:** https://faridaltamash02-ml-classification-project-app-48ftov.streamlit.app/
- **Dataset:** https://archive.ics.uci.edu/dataset/222/bank+marketing

## 1. Problem statement

Predict whether a customer will subscribe to a bank term deposit (`y = yes` or `y = no`) based on information available **before** a marketing call. Compare the following five classifiers on the same held-out test dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM, RBF kernel)

Compare all five using **Accuracy, ROC AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC)**. The positive class for Precision, Recall, and F1 is **`yes`**.

## 2. Dataset description

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Original file:** `data/bank-full.csv` (semicolon-separated)
- **Instances:** 45,211
- **Original columns:** 17 (16 predictor columns and the target `y`)
- **Predictors used:** 15 (six numerical and nine categorical)
- **Target:** `y`, with classes `no` and `yes`
- **Training/test split:** 80:20, stratified by `y`, with `random_state=42`
- **Training set:** 36,168 records
- **Test set:** 9,043 records, stored as `data/test_data.csv` (15 inputs plus `y`)

### Features used

**Numerical:** `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.

**Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.

**Excluded feature:** `duration` (last call length), because it would not be available for a prediction made **before** the call. Using it for pre-call prediction would create a prediction-time information mismatch.

### Class distribution and exploratory analysis

| Target | Count | Share |
|---|---:|---:|
| `no` | 39,922 | 88.30% |
| `yes` | 5,289 | 11.70% |

The data are imbalanced. A majority-class-only baseline would achieve about 88.30% accuracy while identifying no subscribers, so accuracy alone is insufficient.

The original data contain no pandas null values, but several categorical features explicitly contain `unknown`:

| Feature | `unknown` count | Share |
|---|---:|---:|
| `job` | 288 | 0.64% |
| `education` | 1,857 | 4.11% |
| `contact` | 13,020 | 28.80% |
| `poutcome` | 36,959 | 81.75% |

`unknown` was retained as a category. For `pdays`, `-1` means the customer had not previously been contacted and was not treated as a missing numeric value. Numerical distributions show skew, especially for `balance`, `campaign`, and `previous`.

**EDA education observations:** subscription rates were approximately 15.01% for tertiary, 13.57% for unknown, 10.56% for secondary, and 8.63% for primary education. These describe associations and do not establish causality.

EDA charts, including numerical distributions and education/subscription rates, are available in `results/` where included in the repository.

## 3. GitHub repository and project files

**Repository:** https://github.com/faridaltamash02/ml-classification-project

Main project components (file names reflect the working project; check the current repository for exact contents):

```text
ml-classification-project/
├── app.py                         # Streamlit evaluation dashboard
├── explore_data.py                # Exploratory data analysis
├── process.py                     # Train/test split and preprocessing
├── train_logistic.py              # Logistic Regression
├── train_decision_tree.py         # Decision Tree
├── train_knn.py                   # KNN
├── train_naive_bayes.py           # Gaussian Naive Bayes
├── train_svm.py                   # Support Vector Machine
├── evaluate_model.py              # Five-model evaluation (VM execution)
├── requirements.txt
├── README.md
├── data/
│   ├── bank-full.csv
│   └── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
└── results/
    ├── [EDA chart images]
    └── model_comparison.csv
```

## 4. Methodology

1. Load `bank-full.csv` with `sep=";"` and separate the target `y`.
2. Exclude `duration` for pre-call prediction.
3. Create the same stratified 80:20 train/test split for all five classifiers (`random_state=42`).
4. Apply median imputation and `StandardScaler` to numerical columns. Apply most-frequent imputation and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to categorical columns. The latter retains the explicit `unknown` category.
5. Fit preprocessing using training data only, combining preprocessing and classifier in a scikit-learn `Pipeline`. The 15 input features expand to 50 transformed features.
6. Train the five classifiers and save each **complete fitted pipeline** to `model/*.joblib`.
7. Evaluate the **same 9,043 held-out records** for all five models. Compute all six metrics with `yes` as the positive class. Obtain ROC AUC from positive-class probabilities, or from SVM `decision_function` scores.
8. Deploy the Streamlit dashboard to load saved pipelines and evaluate a user-uploaded labeled CSV **without retraining**.

The Gaussian Naive Bayes baseline applies Gaussian feature likelihoods to the one-hot encoded columns as well. Because those binary indicators are not normally distributed and may be dependent, this is a modeling limitation.

## 5. Models used — Final comparison

All values below are from the same held-out test data. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8933** | **0.7717** | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8921 | 0.7054 | **0.6667** | 0.1550 | 0.2515 | 0.2860 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | **0.4641** | **0.4123** | **0.3271** |
| Support Vector Machine | **0.8933** | 0.7183 | 0.6555 | 0.1853 | 0.2889 | 0.3098 |

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Ties for highest accuracy (0.8933) and achieves the highest AUC (0.7717). Positive-class recall (0.1786) is low at the default decision threshold. |
| Decision Tree | Highest positive precision (0.6667), but lowest recall (0.1550) and F1 (0.2515); few subscribers are identified. |
| KNN | Accuracy 0.8874 with recall 0.1975; slightly more positives identified than Logistic Regression or SVM, but lower AUC and precision. |
| Gaussian Naive Bayes | Highest recall (0.4641), F1 (0.4123), and MCC (0.3271). Identifies more subscribers, but has more false positives (833). |
| Support Vector Machine | Ties for highest accuracy (0.8933) and scores AUC 0.7183; recall remains low at 0.1853. |
| **Overall winner for the chosen objective** | **Gaussian Naive Bayes**, provisionally preferred for identifying potential subscribers because it leads on Recall, F1, and MCC. **Logistic Regression** is preferable when stronger ranking (AUC) or fewer false-positive outreach decisions are the priority. |

### Interpretation of the selected winner

The minority class (`yes`) is the outcome of interest in identifying potential term-deposit subscribers. Under an objective that emphasizes finding subscribers, Gaussian Naive Bayes offers the strongest tested positive-class Recall, F1 and MCC. This comes at the cost of reduced Precision (0.3708) and 833 false positives. Model selection depends on the relative business cost of missed subscribers and unnecessary contacts. The chosen winner is based on these baseline test results, not on a separate threshold-optimized validation study.

## 6. Streamlit Community Cloud application

**Live application:** https://faridaltamash02-ml-classification-project-app-48ftov.streamlit.app/

The dashboard includes:

- CSV upload for labeled test data, with required columns and target `y` validation.
- Selection among all five previously trained classifiers.
- Display of Accuracy, AUC, Precision, Recall, F1, and MCC.
- Confusion matrix and detailed classification report.
- Download of evaluation metrics as CSV.
- Model inference using the saved fitted pipeline; **the app does not retrain** on uploaded data.

To verify, upload `data/test_data.csv` and evaluate any model. Gaussian Naive Bayes should return approximately Accuracy **0.845184**, AUC **0.751425**, Recall **0.464083**, and MCC **0.327093**.

## 7. Reproduction and execution

Create and activate a Python virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
```

> **Model compatibility:** The saved models were trained with scikit-learn **1.6.1**, NumPy **2.0.2**, SciPy **1.13.1**, and joblib **1.5.3**. Ensure the deployed/running environment uses compatible versions. Pin or verify these versions in the actual repository `requirements.txt`; a broad scikit-learn range can break model loading.

Evaluate the saved models **without retraining**:

```bash
python evaluate_model.py
```

Launch the Streamlit app:

```bash
python -m streamlit run app.py
```

The saved fitted pipelines are trusted project artifacts. Do not load arbitrary `.joblib` files from untrusted users.

## 8. BITS Virtual Lab execution

The five-model evaluation was also executed within the **BITS PragyaShala Virtual Lab VM** using its Jupyter terminal. The resulting confusion matrices and comparison table matched the local evaluation values. A screenshot of this genuine Virtual Lab execution was captured for inclusion in the assignment submission PDF.

## 9. Submission checklist

- [x] Public classification dataset satisfying minimum instance/feature requirements
- [x] All five required classifiers implemented and trained
- [x] All six required evaluation metrics calculated on the same test set
- [x] Model observations and overall winner documented
- [x] Streamlit application deployed and tested
- [x] Assignment executed within BITS Virtual Lab and screenshot captured
- [ ] Confirm GitHub `main` contains all five `.joblib` files, `data/test_data.csv`, source files, and this final README
- [ ] Confirm `requirements.txt` in GitHub is pinned to the compatible training dependencies
- [ ] Assemble the **single final submission PDF** in the required order: GitHub link, live Streamlit link, BITS Virtual Lab execution screenshot, and complete README content

## 10. Dataset attribution

UCI Machine Learning Repository, *Bank Marketing* dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing. Consult the UCI dataset page for original authorship, citation, and license information.
