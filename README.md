# Bank Marketing Subscription Prediction — End-to-End ML Pipeline


A complete, reproducible machine learning pipeline built on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) to predict whether a customer will subscribe to a term deposit after a telemarketing call. The project covers all four stages of the ML lifecycle: **EDA & Pre-processing → Clustering → Regression → Classification**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Methods & Models](#methods--models)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Design Decisions](#design-decisions)

---

## Project Overview

The core challenge is predicting term deposit subscription from **pre-call information only** - meaning the actual call duration cannot be used as a predictor (it would cause target leakage, since you only know call length after the call ends).

To solve this properly, the pipeline engineers two features before the final classifier sees the data:

1. **Customer Segment** - derived from unsupervised clustering on pre-call features
2. **Predicted Call Duration** - a regression model estimate based on pre-call features and the segment label

These engineered features flow downstream, giving the classifier richer signal without leaking post-call information.

---

## Pipeline Architecture

```
Raw Data (bank-full-updated.csv)
        │
        ▼
┌─────────────────────────────┐
│  Section 1: EDA &           │
│  Pre-processing             │
│  • Feature engineering      │
│  • OHE + StandardScaler ✓   │  ◄── selected approach
│  • Ordinal + MinMaxScaler   │
└─────────────┬───────────────┘
              │  pre-call features
              ▼
┌─────────────────────────────┐
│  Section 2: Clustering      │
│  • K-Means (k=3) ✓          │  ◄── selected approach
│  • Gaussian Mixture Model   │
│  Output: cluster label      │
└─────────────┬───────────────┘
              │  pre-call features + cluster
              ▼
┌─────────────────────────────┐
│  Section 3: Regression      │
│  • Random Forest ✓          │  ◄── selected approach
│  • Ridge Regression         │
│  Output: predicted_duration │
└─────────────┬───────────────┘
              │  pre-call features + cluster + predicted_duration
              ▼
┌─────────────────────────────┐
│  Section 4: Classification  │
│  • Logistic Regression ✓    │  ◄── selected approach
│  • Gradient Boosting        │
│  Output: subscription (y/n) │
└─────────────────────────────┘
```

---

## Dataset

**Source:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**File:** `bank-full-updated.csv`  
**Size:** ~45,000 rows × 18 columns  
**Class imbalance:** ~88.3% No / ~11.7% Yes (7.5:1 ratio)

| Feature | Type | Description |
|---|---|---|
| `age` | numeric | Client age |
| `job` | categorical | Type of job |
| `marital` | categorical | Marital status |
| `education` | categorical | Education level |
| `default` | categorical | Has credit in default? |
| `balance` | numeric | Average yearly balance (€) |
| `housing` | categorical | Has housing loan? |
| `loan` | categorical | Has personal loan? |
| `contact` | categorical | Contact communication type |
| `day` | numeric | Last contact day of month |
| `month` | categorical | Last contact month |
| `duration` | numeric | Last call duration (seconds) - **target for regression, excluded from classification** |
| `campaign` | numeric | Number of contacts in this campaign |
| `pdays` | numeric | Days since last contacted (−1 = never) |
| `previous` | numeric | Contacts before this campaign |
| `poutcome` | categorical | Outcome of previous campaign |
| `y` | binary | **Subscription target (yes/no)** |

**Engineered features (created in this pipeline):**

| Feature | Source | Description |
|---|---|---|
| `was_previously_contacted` | `pdays` | Binary flag: 1 if client was contacted before |
| `cluster` | K-Means (Section 2) | Customer segment label (0, 1, or 2) |
| `predicted_duration` | Random Forest (Section 3) | Model-estimated call duration |

---

## Methods & Models

### Section 1 - EDA & Pre-processing

| Approach | Encoding | Scaling | CV F1 | CV AUC |
|---|---|---|---|---|
| **Approach 1 ✓** | One-Hot Encoding | StandardScaler | 0.376 | 0.766 |
| Approach 2 | Ordinal Encoding | MinMaxScaler | 0.327 | 0.735 |

**Selected:** One-Hot Encoding + StandardScaler — preserves the nominal nature of categorical features and handles outliers in balance/campaign more robustly than min-max scaling.

### Section 2 - Clustering (Customer Segmentation)

| Method | Clusters | Silhouette Score |
|---|---|---|
| **K-Means ✓** | 3 | 0.134 |
| Gaussian Mixture Model | 10 (BIC-optimal) | 0.023 |

**Selected:** K-Means with k=3 - produces interpretable, well-separated segments (subscription rates: 23.1% / 10.0% / 8.3%) and shows clear structure in PCA projections. GMM's BIC kept decreasing to k=10, indicating overfitting to the high-dimensional covariance structure.

### Section 3 - Regression (Call Duration Prediction)

| Method | R² | RMSE | MAE |
|---|---|---|---|
| **Random Forest ✓** | 0.019 | 255.12 | 168.09 |
| Ridge Regression | 0.016 | 255.48 | 168.43 |

**Selected:** Random Forest - marginally outperforms Ridge on all metrics and can capture non-linear feature interactions (e.g. balance × age effects on call length). Both models show low R², which is expected: call duration is largely determined by real-time conversation dynamics unavailable pre-call. Out-of-fold (`cross_val_predict`) predictions are used to prevent leakage into the classification stage.

### Section 4 - Classification (Subscription Prediction)

| Method | AUC-ROC | F1 (Yes) | Precision (Yes) | Recall (Yes) | Accuracy |
|---|---|---|---|---|---|
| **Logistic Regression ✓** | 0.775 | 0.38 | 0.27 | 0.64 | 76% |
| Gradient Boosting | 0.801 | 0.36 | 0.65 | 0.25 | 90% |

**Selected:** Logistic Regression - despite lower overall accuracy, it achieves far higher recall (64% vs 25%) on the minority class. In a telemarketing context, missing a genuine subscriber is more costly than an unnecessary call, so recall is the priority metric. Gradient Boosting's high precision comes at the expense of missing three quarters of potential subscribers.

---

## Key Results

- **Pipeline integrity confirmed:** actual `duration` is never used as a classification feature
- **Leakage prevention:** regression predictions use out-of-fold (`cross_val_predict`) to ensure no row's label is predicted by a model trained on it
- **Engineered feature validation:** `predicted_duration` appears as the 2nd most important feature in Gradient Boosting's importance ranking (10%), confirming the regression stage adds value
- **Cluster utility:** subscription rates differ meaningfully across K-Means segments (8–23%), validating the segmentation step as a useful feature

---

## Project Structure

```
.
├── H_M_Ibnul_Akib_250268289_DG4AML.ipynb   # Main notebook (all code + justification)
├── bank-full-updated.csv                    # Dataset
└── README.md                                # This file
```

The notebook is self-contained and executes end-to-end without manual intervention. It is structured into four clearly labelled sections matching the assessment rubric.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook H_M_Ibnul_Akib_250268289_DG4AML.ipynb
```

> **Note:** The notebook loads the dataset from the working directory. Make sure `bank-full-updated.csv` is in the same folder as the notebook, or update the file path in the first cell.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Tested with Python 3.9+.

---

## Design Decisions

**Why not use call duration as a feature?**  
Call duration is only known after the call ends - using it to predict subscription would be target leakage. The pipeline instead *predicts* duration from pre-call features and uses that estimate.

**Why address class imbalance without resampling?**  
Rather than SMOTE or undersampling, `class_weight='balanced'` is used in Logistic Regression and `scale_pos_weight` is considered for Gradient Boosting. This avoids synthetic data artefacts while still correcting for the 7.5:1 skew.

**Why cross_val_predict for regression outputs?**  
If a model is trained on the full dataset and its predictions used as features in a downstream model, the downstream model sees "memorised" predictions for training rows — a form of data leakage. `cross_val_predict` generates out-of-fold predictions, so each row's predicted duration came from a model that never saw it during training.

---

*Built as part of the DG4AML Applied Machine Learning module at Aston University (2025/26).*
