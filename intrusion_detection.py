"""
=============================================================================
  Network Intrusion Detection System using Machine Learning
  Based on NSL-KDD Dataset Schema (Synthetic Realistic Data)
=============================================================================

  Project    : Cybersecurity — Network Intrusion Detection
  Algorithm  : Random Forest Classifier
  Dataset    : NSL-KDD style (synthetically generated, reproducible)
  Author     : MS Cybersecurity Project
  Python     : 3.8+

  What this project does:
  -----------------------
  Detects whether a network connection is NORMAL or an ATTACK using
  features extracted from network traffic logs — exactly how a real
  Intrusion Detection System (IDS) works.

  The NSL-KDD dataset is the gold standard benchmark for IDS research.
  It contains features like:
    - Duration, protocol type, service, flag
    - Bytes sent/received, login failures, error rates
    - Connection count, host-based traffic statistics

  Pipeline:
    1. Data Generation  (NSL-KDD schema)
    2. Exploratory Data Analysis (EDA)
    3. Data Cleaning & Preprocessing
    4. Feature Engineering & Encoding
    5. Train / Test Split
    6. Model Training  (Random Forest)
    7. Evaluation  (Accuracy, Precision, Recall, F1, AUC-ROC)
    8. Visualization  (Confusion Matrix, Feature Importance, ROC Curve)

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # works without a display (servers / headless)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)              # reproducibility

# Output folder for all saved plots
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  Network Intrusion Detection System — ML Pipeline")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
# We generate a realistic NSL-KDD-style dataset.
# The NSL-KDD dataset is widely used in cybersecurity research.
# Reference: Tavallaee et al. (2009) "A detailed analysis of the KDD CUP 99
#            dataset", IEEE CISDA.
#
# Each row represents ONE network connection (TCP/UDP/ICMP session).
# Features mirror the original 41-feature NSL-KDD schema (subset used here
# for clarity). Statistical distributions are calibrated to match the
# real dataset's known class-conditional properties.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1]  Generating NSL-KDD-style Dataset ...")

N = 5000        # total samples

# Attack labels and their approximate proportions from the real dataset
ATTACK_LABELS = ['neptune', 'smurf', 'pod', 'teardrop',
                 'back', 'land', 'warezclient', 'portsweep', 'ipsweep']
LABEL_POOL    = ['normal'] + ATTACK_LABELS
LABEL_PROBS   = [0.40, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]

labels = np.random.choice(LABEL_POOL, N, p=LABEL_PROBS)

is_normal  = (labels == 'normal')
is_dos     = np.isin(labels, ['neptune', 'smurf', 'pod', 'teardrop', 'back'])
is_probe   = np.isin(labels, ['portsweep', 'ipsweep'])
is_r2l     = np.isin(labels, ['warezclient'])
is_land    = (labels == 'land')

# --- Feature generation (calibrated per attack category) ---

# duration: normal connections last longer
duration = np.where(is_normal,
                    np.random.exponential(5, N),
                    np.random.exponential(0.5, N)).astype(int)

# protocol type: tcp most common, icmp heavily used in DoS (smurf)
protocol_type = np.where(
    np.isin(labels, ['smurf', 'pod']),
    'icmp',
    np.random.choice(['tcp', 'udp', 'icmp'], N, p=[0.65, 0.25, 0.10])
)

# service
service = np.where(
    is_normal,
    np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns', 'other'],
                     N, p=[0.40, 0.10, 0.10, 0.10, 0.10, 0.20]),
    np.random.choice(['http', 'private', 'domain_u', 'smtp', 'other'],
                     N, p=[0.20, 0.30, 0.20, 0.10, 0.20])
)

# flag: SF = normal completion; S0 = no reply (DoS indicator)
flag = np.where(
    is_dos,
    np.random.choice(['S0', 'REJ', 'RSTO'], N, p=[0.60, 0.25, 0.15]),
    np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'SH'],
                     N, p=[0.70, 0.10, 0.08, 0.07, 0.05])
)

# bytes sent from source
src_bytes = np.where(is_normal,
                     np.random.randint(200, 60000, N),
                     np.random.randint(0, 600, N))

# bytes sent from destination
dst_bytes = np.where(is_normal,
                     np.random.randint(200, 60000, N),
                     np.random.randint(0, 100, N))

# land = 1 means src and dst IP/port are same (land attack)
land = np.where(is_land, 1, 0)

# wrong_fragment: packet fragmentation errors (pod, teardrop)
wrong_fragment = np.where(
    np.isin(labels, ['pod', 'teardrop']),
    np.random.randint(1, 6, N),
    np.zeros(N, int)
)

# urgent packets
urgent = np.random.choice([0, 1, 2], N, p=[0.95, 0.04, 0.01])

# hot: number of "hot" indicators (sensitive accesses)
hot = np.where(is_normal, np.random.randint(0, 30, N), np.random.randint(0, 5, N))

# failed logins
num_failed_logins = np.where(
    np.isin(labels, ['neptune', 'warezclient']),
    np.random.randint(1, 7, N),
    np.zeros(N, int)
)

# logged_in: successful login (1 = yes)
logged_in = np.where(
    is_normal,
    np.random.choice([0, 1], N, p=[0.15, 0.85]),
    np.random.choice([0, 1], N, p=[0.75, 0.25])
)

# count: number of connections to same host in past 2 seconds
count = np.where(
    is_dos | is_probe,
    np.random.randint(200, 512, N),
    np.random.randint(1, 100, N)
)

# srv_count: connections to same service in past 2 seconds
srv_count = np.where(
    is_dos,
    np.random.randint(200, 512, N),
    np.random.randint(1, 150, N)
)

# serror_rate: % connections with SYN errors (high in DoS)
serror_rate = np.where(
    is_dos,
    np.random.uniform(0.75, 1.0, N),
    np.random.uniform(0.0, 0.15, N)
)

# dst_host_count: connections to same destination host
dst_host_count = np.where(
    is_probe,
    np.random.randint(200, 256, N),
    np.random.randint(1, 100, N)
)

# dst_host_srv_count
dst_host_srv_count = np.where(
    is_dos,
    np.random.randint(200, 256, N),
    np.random.randint(1, 200, N)
)

# ─── Assemble DataFrame ───
df = pd.DataFrame({
    'duration'            : duration,
    'protocol_type'       : protocol_type,
    'service'             : service,
    'flag'                : flag,
    'src_bytes'           : src_bytes,
    'dst_bytes'           : dst_bytes,
    'land'                : land,
    'wrong_fragment'      : wrong_fragment,
    'urgent'              : urgent,
    'hot'                 : hot,
    'num_failed_logins'   : num_failed_logins,
    'logged_in'           : logged_in,
    'count'               : count,
    'srv_count'           : srv_count,
    'serror_rate'         : serror_rate,
    'dst_host_count'      : dst_host_count,
    'dst_host_srv_count'  : dst_host_srv_count,
    'attack_label'        : labels,            # multi-class label
})

# Binary target: normal vs attack (used for main classification)
df['target'] = df['attack_label'].apply(lambda x: 0 if x == 'normal' else 1)

# Save raw dataset
df.to_csv("raw_dataset.csv", index=False)
print(f"  Dataset shape        : {df.shape}")
print(f"  Normal connections   : {(df['target']==0).sum()}")
print(f"  Attack connections   : {(df['target']==1).sum()}")
print(f"  Raw data saved to    : raw_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2]  Exploratory Data Analysis ...")

print("\n  --- Dataset Info ---")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Numeric feature statistics:")
print(df.describe().to_string())

print(f"\n  Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  None — dataset is complete.")

print(f"\n  Attack label breakdown:")
print(df['attack_label'].value_counts().to_string())

# ── Plot 1: Class distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class Distribution", fontsize=14, fontweight='bold')

# Binary
counts = df['target'].value_counts()
axes[0].bar(['Normal (0)', 'Attack (1)'], counts.values,
            color=['#2196F3', '#F44336'], edgecolor='black', width=0.5)
axes[0].set_title("Binary: Normal vs Attack")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# Multi-class
ml = df['attack_label'].value_counts()
axes[1].barh(ml.index, ml.values, color='#455A64', edgecolor='black')
axes[1].set_title("Attack Type Breakdown")
axes[1].set_xlabel("Count")
for i, v in enumerate(ml.values):
    axes[1].text(v + 10, i, str(v), va='center')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/01_class_distribution.png")

# ── Plot 2: Protocol / Service / Flag distributions ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight='bold')

for ax, col, title in zip(axes,
                           ['protocol_type', 'service', 'flag'],
                           ['Protocol Type', 'Service', 'Flag']):
    vc = df[col].value_counts()
    ax.bar(vc.index, vc.values, color='#1976D2', edgecolor='black')
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_categorical_features.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/02_categorical_features.png")

# ── Plot 3: Numeric feature distributions (Normal vs Attack) ─────────────────
num_features = ['duration', 'src_bytes', 'dst_bytes', 'count',
                'srv_count', 'serror_rate', 'dst_host_count']

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
fig.suptitle("Feature Distributions: Normal vs Attack", fontsize=14, fontweight='bold')

for i, feat in enumerate(num_features):
    ax = axes[i]
    normal_data = df.loc[df['target'] == 0, feat]
    attack_data = df.loc[df['target'] == 1, feat]
    ax.hist(normal_data, bins=30, alpha=0.6, color='#2196F3', label='Normal', density=True)
    ax.hist(attack_data, bins=30, alpha=0.6, color='#F44336', label='Attack', density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

axes[-1].axis('off')   # hide the empty 8th subplot
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/03_feature_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3]  Data Cleaning ...")

# 3.1 Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"  Duplicates removed   : {before - after}")

# 3.2 Handle missing values (none expected, but we check and fill defensively)
null_counts = df.isnull().sum()
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  Filled missing in '{col}' with mode")
        else:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  Filled missing in '{col}' with median")
if null_counts.sum() == 0:
    print("  No missing values found — dataset is clean.")

# 3.3 Remove impossible/outlier values
# serror_rate must be in [0, 1]
df = df[(df['serror_rate'] >= 0) & (df['serror_rate'] <= 1)]
# count must be positive
df = df[df['count'] >= 0]
# bytes cannot be negative
df = df[(df['src_bytes'] >= 0) & (df['dst_bytes'] >= 0)]
print(f"  Rows after cleaning  : {len(df)}")

# 3.4 Clip extreme outliers using IQR (keep within 1.5 × IQR fence)
numeric_cols = ['duration', 'src_bytes', 'dst_bytes',
                'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
print("  Outliers clipped (IQR method) for numeric features.")

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print(f"  Cleaned data saved to: cleaned_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4]  Feature Engineering & Encoding ...")

# 4.1 Encode categorical features with LabelEncoder
#     (suitable for tree-based models like Random Forest)
cat_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}' : {list(le.classes_)}")

# 4.2 Feature selection — drop label columns, keep features
FEATURE_COLS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'count', 'srv_count',
    'serror_rate', 'dst_host_count', 'dst_host_srv_count'
]

X = df[FEATURE_COLS].copy()
y = df['target'].copy()

print(f"\n  Feature matrix shape : {X.shape}")
print(f"  Target distribution  : {dict(y.value_counts())}")

# 4.3 Feature correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/04_correlation_matrix.png")

# 4.4 Scale features (important for distance-based models; good practice for RF too)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5]  Train / Test Split ...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,          # 80 % training, 20 % testing
    random_state=42,
    stratify=y               # preserve class ratio
)

print(f"  Training samples     : {len(X_train)}")
print(f"  Testing  samples     : {len(X_test)}")
print(f"  Train class balance  : {dict(y_train.value_counts())}")
print(f"  Test  class balance  : {dict(y_test.value_counts())}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 6]  Training Random Forest Classifier ...")
print("  (This may take a few seconds ...)\n")

# Random Forest is the standard choice for IDS because:
#   - Handles mixed numeric/categorical features
#   - Robust to outliers (uses decision trees internally)
#   - Provides feature importance natively
#   - Excellent accuracy on tabular data
#   - Reduces overfitting via bagging of many trees

rf_model = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=15,            # limit depth to avoid overfitting
    min_samples_split=5,     # minimum samples to split a node
    min_samples_leaf=2,      # minimum samples at leaf nodes
    max_features='sqrt',     # features considered per split (sqrt rule)
    class_weight='balanced', # handles class imbalance automatically
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

rf_model.fit(X_train, y_train)
print("  Training complete.")

# 5-fold cross-validation on training set (gives honest performance estimate)
print("\n  Running 5-fold Cross-Validation on training set ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1')
print(f"  CV F1 scores         : {[round(s, 4) for s in cv_scores]}")
print(f"  CV F1 Mean ± Std     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 7]  Evaluating Model on Test Set ...")

y_pred       = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]   # probability of attack

# Core metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)

print("\n  ┌─────────────────────────────────────────┐")
print(  "  │         EVALUATION RESULTS              │")
print(  "  ├──────────────────────┬──────────────────┤")
print(f"  │  Accuracy            │  {acc:.4f}          │")
print(f"  │  Precision           │  {prec:.4f}          │")
print(f"  │  Recall              │  {rec:.4f}          │")
print(f"  │  F1-Score            │  {f1:.4f}          │")
print(f"  │  AUC-ROC             │  {auc:.4f}          │")
print(  "  └──────────────────────┴──────────────────┘")

print("\n  Detailed Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Normal (0)', 'Attack (1)']))

# ── Plot 4: Confusion Matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix — Random Forest\n"
             f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}",
             fontsize=12, fontweight='bold')
ax.set_ylabel("Actual Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)

# Annotate TN, FP, FN, TP
labels_cm = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.75, labels_cm[i][j],
                ha='center', va='center', fontsize=9,
                color='grey', style='italic')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {OUTPUT_DIR}/05_confusion_matrix.png")

# ── Plot 5: ROC Curve ────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#1976D2', lw=2.5,
        label=f'Random Forest (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--',
        label='Random Classifier (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.10, color='#1976D2')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve — Intrusion Detection", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/06_roc_curve.png")

# ── Plot 6: Feature Importance ───────────────────────────────────────────────
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature'   : FEATURE_COLS,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#1976D2' if v > feat_imp_df['Importance'].median()
          else '#90CAF9' for v in feat_imp_df['Importance']]
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
               color=colors, edgecolor='white')
for bar, val in zip(bars, feat_imp_df['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel("Gini Importance", fontsize=12)
ax.set_title("Feature Importance — Random Forest\n"
             "(Higher = More Discriminative for Attack Detection)",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/07_feature_importance.png")

# ── Plot 7: Cross-Validation Scores ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax.bar(folds, cv_scores, color='#1976D2', edgecolor='black', width=0.5)
ax.axhline(cv_scores.mean(), color='red', linestyle='--', lw=2,
           label=f'Mean F1 = {cv_scores.mean():.4f}')
ax.set_ylim([0.85, 1.01])
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("5-Fold Cross-Validation F1 Scores", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_cross_validation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/08_cross_validation.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — LIVE PREDICTION DEMO
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 8]  Live Prediction Demo ...")
print("  (Simulating 5 new incoming network connections)\n")

# Manually craft 3 attack and 2 normal samples for demo
demo_raw = pd.DataFrame([
    # --- Normal connections ---
    {  # Typical HTTP session
        'duration': 5, 'protocol_type': 'tcp', 'service': 'http',
        'flag': 'SF', 'src_bytes': 45000, 'dst_bytes': 38000,
        'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 12,
        'num_failed_logins': 0, 'logged_in': 1, 'count': 10,
        'srv_count': 9, 'serror_rate': 0.02, 'dst_host_count': 30,
        'dst_host_srv_count': 28
    },
    {  # Typical SSH session
        'duration': 120, 'protocol_type': 'tcp', 'service': 'ssh',
        'flag': 'SF', 'src_bytes': 12000, 'dst_bytes': 9000,
        'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 5,
        'num_failed_logins': 0, 'logged_in': 1, 'count': 4,
        'srv_count': 3, 'serror_rate': 0.0, 'dst_host_count': 8,
        'dst_host_srv_count': 7
    },
    # --- Attack connections ---
    {  # Neptune (SYN Flood DoS)
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
        'flag': 'S0', 'src_bytes': 0, 'dst_bytes': 0,
        'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
        'num_failed_logins': 5, 'logged_in': 0, 'count': 510,
        'srv_count': 510, 'serror_rate': 0.99, 'dst_host_count': 255,
        'dst_host_srv_count': 250
    },
    {  # Smurf (ICMP amplification DoS)
        'duration': 0, 'protocol_type': 'icmp', 'service': 'other',
        'flag': 'SF', 'src_bytes': 520, 'dst_bytes': 0,
        'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
        'num_failed_logins': 0, 'logged_in': 0, 'count': 511,
        'srv_count': 511, 'serror_rate': 0.0, 'dst_host_count': 255,
        'dst_host_srv_count': 255
    },
    {  # Land attack (src == dst)
        'duration': 0, 'protocol_type': 'tcp', 'service': 'smtp',
        'flag': 'S0', 'src_bytes': 0, 'dst_bytes': 0,
        'land': 1, 'wrong_fragment': 3, 'urgent': 0, 'hot': 0,
        'num_failed_logins': 3, 'logged_in': 0, 'count': 2,
        'srv_count': 2, 'serror_rate': 0.95, 'dst_host_count': 1,
        'dst_host_srv_count': 1
    },
])

# Encode categorical columns using the same encoders
for col in cat_cols:
    le = encoders[col]
    # Handle unseen labels safely
    demo_raw[col] = demo_raw[col].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )

# Apply the same scaler
demo_scaled = scaler.transform(demo_raw[FEATURE_COLS])

# Predict
demo_pred  = rf_model.predict(demo_scaled)
demo_proba = rf_model.predict_proba(demo_scaled)[:, 1]

ground_truth = ['Normal', 'Normal', 'Attack', 'Attack', 'Attack']
print(f"  {'#':<4} {'Connection':<26} {'True Label':<12} {'Predicted':<12} {'Attack Prob'}")
print("  " + "-" * 65)
conn_names = ['HTTP Session', 'SSH Session', 'Neptune (SYN Flood)',
              'Smurf (ICMP DoS)', 'Land Attack']
for i, (name, gt, pred, prob) in enumerate(
        zip(conn_names, ground_truth, demo_pred, demo_proba)):
    pred_label = "Attack" if pred == 1 else "Normal"
    correct    = "✓" if gt == pred_label else "✗"
    print(f"  {i+1:<4} {name:<26} {gt:<12} {pred_label:<12} {prob:.4f}  {correct}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  FINAL SUMMARY REPORT")
print("=" * 65)
print(f"""
  Dataset         : NSL-KDD style (synthetic, 5000 samples)
  Task            : Binary Classification (Normal vs Attack)
  Model           : Random Forest (100 trees, max_depth=15)

  ── Training ──────────────────────────────────────────
  Train samples   : {len(X_train)}
  Test  samples   : {len(X_test)}
  CV F1 (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

  ── Test Set Performance ──────────────────────────────
  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)
  Precision       : {prec:.4f}
  Recall          : {rec:.4f}
  F1-Score        : {f1:.4f}
  AUC-ROC         : {auc:.4f}

  ── Top 3 Most Important Features ────────────────────
""")
top3 = feat_imp_df.tail(3)[::-1]
for rank, (_, row) in enumerate(top3.iterrows(), 1):
    print(f"  {rank}. {row['Feature']:<25}  Importance: {row['Importance']:.4f}")

print(f"""
  ── Output Files ─────────────────────────────────────
  raw_dataset.csv
  cleaned_dataset.csv
  {OUTPUT_DIR}/01_class_distribution.png
  {OUTPUT_DIR}/02_categorical_features.png
  {OUTPUT_DIR}/03_feature_distributions.png
  {OUTPUT_DIR}/04_correlation_matrix.png
  {OUTPUT_DIR}/05_confusion_matrix.png
  {OUTPUT_DIR}/06_roc_curve.png
  {OUTPUT_DIR}/07_feature_importance.png
  {OUTPUT_DIR}/08_cross_validation.png
""")
print("=" * 65)
print("  Pipeline complete. All outputs saved successfully.")
print("=" * 65)
