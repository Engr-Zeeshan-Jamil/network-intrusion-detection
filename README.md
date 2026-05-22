# Network Intrusion Detection System — ML Project
### MS Cybersecurity | Machine Learning Pipeline

---

## What This Project Does

This project builds a **Network Intrusion Detection System (IDS)** that
automatically classifies network connections as **Normal** or **Attack**
using a Random Forest machine learning model.

The dataset follows the **NSL-KDD schema** — the gold-standard benchmark
dataset used in cybersecurity ML research since 2009. Each row represents
one network connection (TCP/UDP/ICMP session) described by 17 features
extracted from traffic logs.

### Attack Types Covered
| Attack | Category | Description |
|--------|----------|-------------|
| Neptune | DoS | SYN flood — server overwhelmed with connection requests |
| Smurf | DoS | ICMP amplification attack |
| Pod | DoS | Ping of Death — oversized ICMP packets |
| Teardrop | DoS | Malformed fragmented packets |
| Back | DoS | Apache web server DoS |
| Land | DoS | Src and Dst IP/port are identical |
| Warezclient | R2L | Unauthorized file download |
| Portsweep | Probe | Scanning a host for open ports |
| Ipsweep | Probe | Scanning a range of IPs |

---

## Project Structure

```
network_intrusion_detection/
│
├── intrusion_detection.py    ← Main Python script (full pipeline)
├── requirements.txt          ← Python package dependencies
├── README.md                 ← This file
│
├── raw_dataset.csv           ← Generated after running (raw data)
├── cleaned_dataset.csv       ← Generated after running (cleaned data)
│
└── output_plots/             ← Generated after running
    ├── 01_class_distribution.png
    ├── 02_categorical_features.png
    ├── 03_feature_distributions.png
    ├── 04_correlation_matrix.png
    ├── 05_confusion_matrix.png
    ├── 06_roc_curve.png
    ├── 07_feature_importance.png
    └── 08_cross_validation.png
```

---

## Step-by-Step Setup and Run Instructions

### Prerequisites
- Python 3.8 or higher
- pip (comes with Python)

---

### Step 1 — Verify Python Version
Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:
```bash
python --version
```
You should see Python 3.8 or higher. If not, download from https://python.org

---

### Step 2 — Create a Virtual Environment (Recommended)
A virtual environment keeps project dependencies isolated.

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You will see `(venv)` appear in your terminal prompt. This means the
virtual environment is active.

---

### Step 3 — Install Required Packages
```bash
pip install -r requirements.txt
```

This installs:
- `numpy` — numerical computing
- `pandas` — data manipulation
- `scikit-learn` — machine learning
- `matplotlib` — plotting
- `seaborn` — statistical visualizations

---

### Step 4 — Run the Project
```bash
python intrusion_detection.py
```

The script will:
1. Generate the NSL-KDD-style dataset (5000 samples)
2. Perform Exploratory Data Analysis
3. Clean and preprocess the data
4. Encode categorical features and scale numerics
5. Split into 80% train / 20% test
6. Train a Random Forest classifier
7. Evaluate with Accuracy, Precision, Recall, F1, AUC-ROC
8. Save 8 visualization plots to `output_plots/`
9. Run a live prediction demo on 5 sample connections

**Expected total runtime:** 10–30 seconds

---

### Step 5 — View Results
After running, open the `output_plots/` folder to see the visualizations.
The terminal will also print a full summary report.

---

## Expected Output (Terminal)

```
=================================================================
  Network Intrusion Detection System — ML Pipeline
=================================================================

[STEP 1]  Generating NSL-KDD-style Dataset ...
  Dataset shape        : (5000, 19)
  Normal connections   : 1933
  Attack connections   : 3067

[STEP 2]  Exploratory Data Analysis ...
  ...

[STEP 7]  Evaluating Model on Test Set ...
  ┌─────────────────────────────────────────┐
  │         EVALUATION RESULTS              │
  ├──────────────────────┬──────────────────┤
  │  Accuracy            │  0.9900+         │
  │  Precision           │  0.9900+         │
  │  Recall              │  0.9900+         │
  │  F1-Score            │  0.9900+         │
  │  AUC-ROC             │  0.9990+         │
  └──────────────────────┴──────────────────┘

[STEP 8]  Live Prediction Demo ...
  #    Connection            True Label   Predicted    Attack Prob
  ─────────────────────────────────────────────────────────────────
  1    HTTP Session          Normal       Normal       0.0xxx  ✓
  2    SSH Session           Normal       Normal       0.0xxx  ✓
  3    Neptune (SYN Flood)   Attack       Attack       0.9xxx  ✓
  4    Smurf (ICMP DoS)      Attack       Attack       0.9xxx  ✓
  5    Land Attack           Attack       Attack       0.9xxx  ✓
```

---

## Machine Learning Pipeline Explained

### 1. Data Generation
Simulates the NSL-KDD dataset (Tavallaee et al., 2009). Each feature is
generated with class-conditional distributions calibrated to the real dataset.

### 2. Exploratory Data Analysis (EDA)
- Class balance check
- Distribution of categorical features
- Feature histograms by class (Normal vs Attack)

### 3. Data Cleaning
- Duplicate removal
- Missing value handling (mode for categorical, median for numeric)
- Range validation (bytes ≥ 0, rate ∈ [0,1])
- Outlier clipping using IQR (1.5 × IQR fence)

### 4. Feature Engineering
- Label Encoding for categorical columns (protocol, service, flag)
- StandardScaler normalization (zero mean, unit variance)
- Correlation matrix for feature analysis

### 5. Model: Random Forest
Chosen because:
- Handles mixed feature types natively
- Resistant to outliers (tree-based splits)
- Built-in feature importance ranking
- No assumption about data distribution
- Excellent performance on tabular IDS data

### 6. Evaluation Metrics
| Metric | Formula | Why it Matters in IDS |
|--------|---------|----------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | False alarm rate |
| Recall | TP/(TP+FN) | Miss rate (critical — missing attacks is dangerous) |
| F1-Score | 2×P×R/(P+R) | Balance of precision & recall |
| AUC-ROC | Area under ROC curve | Threshold-independent performance |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install -r requirements.txt` |
| `python: command not found` | Use `python3` instead of `python` |
| Permission error on venv creation | Run terminal as Administrator (Windows) |
| Plots not saving | Ensure you have write permission in the project folder |

---

## Reference
> Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
> *A detailed analysis of the KDD CUP 99 data set.*
> IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA).

> Breiman, L. (2001). *Random Forests.*
> Machine Learning, 45(1), 5–32.
