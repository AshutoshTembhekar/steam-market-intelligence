import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA INGESTION & HEADER CORRECTION
# ==========================================
COLUMNS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU", "Required age", "Price",
    "Discount", "DLC count", "About the game", "Supported languages", "Full audio languages",
    "Reviews", "Header image", "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative", "Score rank",
    "Achievements", "Recommendations", "Notes", "Average playtime forever",
    "Average playtime two weeks", "Median playtime forever", "Median playtime two weeks",
    "Developers", "Publishers", "Categories", "Genres", "Tags", "Screenshots", "Movies",
]

def find_games_csv():
    candidates = ["games.csv", "../input/games.csv", "/mnt/user-data/uploads/games.csv"]
    candidates += glob.glob("/kaggle/input/**/games.csv", recursive=True)
    for path in candidates:
        if os.path.exists(path):
            return path
    return "games.csv"

data_path = find_games_csv()
df = pd.read_csv(data_path, header=0, names=COLUMNS)
print(f"Loaded {len(df):,} rows from {data_path}")

# ==========================================
# 2. FEATURE EXTRACTION & TEMPORAL SPLIT
# ==========================================
def count_list(series):
    return series.fillna("[]").str.count("'") // 2

def count_csv(series):
    s = series.fillna("")
    return np.where(s.eq(""), 0, s.str.count(",") + 1)

df["release_date"] = pd.to_datetime(df["Release date"], format="mixed", errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
df["total_reviews"] = df["Positive"] + df["Negative"]

# Target: 1 if >= 100 reviews, 0 otherwise
df["hit"] = (df["total_reviews"] >= 100).astype(int)

# Strictly pre-launch features (available before release)
df["price"] = df["Price"].clip(lower=0, upper=200)
df["is_free"] = (df["Price"] == 0).astype(int)
df["n_languages"] = count_list(df["Supported languages"])
df["n_audio"] = count_list(df["Full audio languages"])
df["n_platforms"] = df[["Windows", "Mac", "Linux"]].sum(axis=1)
df["multi_platform"] = ((df["Mac"]) | (df["Linux"])).astype(int)
df["has_website"] = df["Website"].notna().astype(int)
df["has_support_email"] = df["Support email"].notna().astype(int)
df["desc_len"] = df["About the game"].fillna("").str.len()
df["n_screenshots"] = count_csv(df["Screenshots"])
df["n_achievements"] = df["Achievements"].clip(lower=0, upper=1000)

# Binary flags for key quality categories
categories_str = df["Categories"].fillna("")
df["has_steam_cloud"] = categories_str.str.contains("Steam Cloud", regex=False).astype(int)
df["has_controller"] = categories_str.str.contains("Controller", regex=False).astype(int)
df["has_achievements"] = (df["Achievements"] > 0).astype(int)

# Filter to mature cohort (2014-2022)
cohort = df[df["release_year"].between(2014, 2022)].dropna(subset=["release_year"]).copy()

# Temporal split: Train (2014-2020), Test (2021-2022)
train_df = cohort[cohort["release_year"].between(2014, 2020)].copy()
test_df = cohort[cohort["release_year"].between(2021, 2022)].copy()

features = [
    "price", "is_free", "n_languages", "n_audio", "n_platforms", "multi_platform",
    "has_website", "has_support_email", "desc_len", "n_screenshots", "n_achievements",
    "has_steam_cloud", "has_controller", "has_achievements", "release_month"
]

X_train, y_train = train_df[features], train_df["hit"]
X_test, y_test = test_df[features], test_df["hit"]

base_rate_test = y_test.mean()
print(f"Train samples (2014-2020): {len(X_train):,} | Hit rate: {y_train.mean():.1%}")
print(f"Test samples (2021-2022):  {len(X_test):,} | Hit rate: {base_rate_test:.1%}")

# ==========================================
# 3. PIPELINE SETUP
# ==========================================
numeric_features = [
    "price", "n_languages", "n_audio", "desc_len", "n_screenshots", "n_achievements"
]
binary_features = [f for f in features if f not in numeric_features]

preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("bin", "passthrough", binary_features),
    ]
)

# Model 1: Logistic Regression
lr_model = Pipeline([
    ("prep", preprocessor_lr),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# Model 2: Histogram Gradient Boosting (natively supports unscaled features & NaNs)
hgb_model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
print("\n--- Training Models ---")
lr_model.fit(X_train, y_train)
hgb_model.fit(X_train, y_train)

# Predictions (probability of reaching >= 100 reviews)
prob_lr = lr_model.predict_proba(X_test)[:, 1]
prob_hgb = hgb_model.predict_proba(X_test)[:, 1]

def evaluate_predictions(y_true, y_prob, name):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    
    # Precision at Top 10% highest predicted scores
    top_10_cutoff = np.percentile(y_prob, 90)
    top_10_indices = y_prob >= top_10_cutoff
    prec_top_10 = y_true[top_10_indices].mean()
    lift_top_10 = prec_top_10 / y_true.mean()
    
    print(f"\n[{name}]")
    print(f"  ROC-AUC:               {roc:.3f}")
    print(f"  PR-AUC:                {pr:.3f} (Baseline: {y_true.mean():.3f})")
    print(f"  Precision @ Top 10%:   {prec_top_10:.1%}")
    print(f"  Lift @ Top 10%:        {lift_top_10:.2f}x")
    return roc, pr

evaluate_predictions(y_test, prob_lr, "Logistic Regression")
evaluate_predictions(y_test, prob_hgb, "HistGradientBoosting")

# ==========================================
# 5. PERMUTATION FEATURE IMPORTANCE
# ==========================================
print("\nComputing Permutation Importance for HistGradientBoosting (Test Set)...")
perm_res = permutation_importance(
    hgb_model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": perm_res.importances_mean,
    "Std": perm_res.importances_std
}).sort_values("Importance", ascending=False)

print("\nTop 5 Pre-Launch Traction Drivers:")
print(importance_df.head(5).to_string(index=False))

# ==========================================
# 6. EVALUATION PLOTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_hgb, tpr_hgb, _ = roc_curve(y_test, prob_hgb)

axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, prob_lr):.2f})", color="#1f77d0")
axes[0].plot(fpr_hgb, tpr_hgb, label=f"HistGradientBoosting (AUC = {roc_auc_score(y_test, prob_hgb):.2f})", color="#7a5cff", lw=2)
axes[0].plot([0, 1], [0, 1], color="grey", ls="--", label="Random Chance (0.50)")
axes[0].set_title("ROC Curves (Test Cohort 2021-2022)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# Precision-Recall Curves
prec_lr, rec_lr, _ = precision_recall_curve(y_test, prob_lr)
prec_hgb, rec_hgb, _ = precision_recall_curve(y_test, prob_hgb)

axes[1].plot(rec_lr, prec_lr, label=f"Logistic Regression (PR-AUC = {average_precision_score(y_test, prob_lr):.2f})", color="#1f77d0")
axes[1].plot(rec_hgb, prec_hgb, label=f"HistGradientBoosting (PR-AUC = {average_precision_score(y_test, prob_hgb):.2f})", color="#7a5cff", lw=2)
axes[1].axhline(base_rate_test, color="grey", ls="--", label=f"Baseline Rate ({base_rate_test:.1%})")
axes[1].set_title("Precision-Recall Curves (Test Cohort)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("steam_model_benchmark.png", dpi=300, bbox_inches="tight")
print("\nSaved evaluation curves to 'steam_model_benchmark.png'")
plt.show()