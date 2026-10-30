import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, PURPLE, PINK, TEAL, AMBER = "#1f77d0", "#7a5cff", "#e5508f", "#14b8a6", "#f2a93b"

# ==========================================
# 2. DATA LOADING & HEADER REPAIR
# ==========================================
# The raw CSV header lists 39 names, but rows contain 40 values due to DiscountDLC count.
# Providing the explicit 40-column list fixes the column misalignment after Price.
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
    for p in candidates:
        if os.path.exists(p):
            return p
    return "games.csv"

DATA_PATH = find_games_csv()
df = pd.read_csv(DATA_PATH, header=0, names=COLUMNS)
print(f"Loaded: {DATA_PATH}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ==========================================
# 3. DATA QUALITY AUDIT
# ==========================================
print("\n--- DATA QUALITY AUDIT ---")
print("Duplicate AppIDs:", df["AppID"].duplicated().sum())
print("Repeated names (playtests/demos):", df["Name"].duplicated().sum())

total_reviews_raw = df["Positive"] + df["Negative"]
zero_share = pd.Series({
    "Price = 0 (free)": (df["Price"] == 0).mean(),
    "Zero reviews": (total_reviews_raw == 0).mean(),
    "Owners range '0 - 0'": (df["Estimated owners"] == "0 - 0").mean(),
    "Owners range '0 - 20000'": (df["Estimated owners"] == "0 - 20000").mean(),
    "Metacritic score present": (df["Metacritic score"] > 0).mean(),
    "Required age > 0": (df["Required age"] > 0).mean(),
    "No trailer data (Movies empty)": df["Movies"].isna().mean(),
}) * 100
print("\nCatalog Zero-Inflation & Missingness:")
print(zero_share.round(2))

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
print("\n--- RUNNING FEATURE ENGINEERING ---")
def count_list(s):
    return s.fillna("[]").str.count("'") // 2

def count_csv(s):
    s = s.fillna("")
    return np.where(s.eq(""), 0, s.str.count(",") + 1)

d = df.copy()
d["release_date"] = pd.to_datetime(d["Release date"], format="mixed", errors="coerce")
d["release_year"] = d["release_date"].dt.year
d["total_reviews"] = d["Positive"] + d["Negative"]
d["pos_ratio"] = d["Positive"] / d["total_reviews"].replace(0, np.nan)
d["price"] = d["Price"]
d["is_free"] = (d["Price"] == 0).astype(int)

d["n_languages"] = count_list(d["Supported languages"])
d["n_platforms"] = d[["Windows", "Mac", "Linux"]].sum(axis=1)
d["multi_platform"] = ((d["Mac"]) | (d["Linux"])).astype(int)
d["desc_len"] = d["About the game"].fillna("").str.len()
d["n_screenshots"] = count_csv(d["Screenshots"])

SNAPSHOT = d.loc[d["total_reviews"] > 0, "release_date"].max()
d["released"] = d["release_date"] <= SNAPSHOT
d["hit"] = (d["total_reviews"] >= 100).astype(int)
released = d[d["released"]].copy()

# ==========================================
# 5. DESCRIPTIVE STATS TABLE
# ==========================================
print("\n--- DESCRIPTIVE STATISTICS (RELEASED TITLES) ---")
numeric_cols = ["Price", "total_reviews", "pos_ratio", "Achievements", "n_languages"]
desc = released[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
desc["IQR"] = desc["75%"] - desc["25%"]
desc["Skewness"] = released[numeric_cols].skew()
desc["Kurtosis"] = released[numeric_cols].kurt()
print(desc[["mean", "std", "50%", "IQR", "Skewness", "Kurtosis"]].round(2))

# ==========================================
# 6. STATISTICAL TESTS (MATURE COHORT 2014-2022)
# ==========================================
mature = released[released["release_year"].between(2014, 2022)].copy()

# 6.1 Mann-Whitney U: Free vs. Paid
free = mature[mature["is_free"] == 1]["total_reviews"]
paid_ = mature[mature["is_free"] == 0]["total_reviews"]
u_stat, p_mw = stats.mannwhitneyu(free, paid_, alternative="two-sided")
prob_sup = u_stat / (len(free) * len(paid_))
print(f"\nMann-Whitney U (Reviews - Free vs Paid): p = {p_mw:.3e}, Prob of Superiority = {prob_sup:.3f}")

# 6.2 Multi-Platform Reach Odds Ratio
tab = pd.crosstab(mature["multi_platform"], mature["hit"])
chi2, p_chi, _, _ = stats.chi2_contingency(tab)
odds_ratio = (tab.loc[1, 1] * tab.loc[0, 0]) / (tab.loc[1, 0] * tab.loc[0, 1])
print(f"Platform Reach Odds Ratio: {odds_ratio:.2f} (Chi2 p = {p_chi:.3e})")

# 6.3 Lorenz Curve & Gini Coefficient
rev = np.sort(released.loc[released["total_reviews"] > 0, "total_reviews"].values)
cum = np.cumsum(rev) / rev.sum()
x_ = np.arange(1, len(rev) + 1) / len(rev)
trap = getattr(np, "trapezoid", None) or np.trapz
gini = 1 - 2 * trap(cum, x_)
print(f"Gini Coefficient of Total Reviews: {gini:.3f}")

# ==========================================
# 7. MULTI-PANEL VISUALIZATION DASHBOARD
# ==========================================
print("\nGenerating visual dashboard...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Lorenz Curve (Inequality)
axes[0].plot(x_, cum, color=PURPLE, lw=2, label=f"Gini = {gini:.3f}")
axes[0].plot([0, 1], [0, 1], color="grey", ls="--", label="Equality")
axes[0].set_title("Lorenz Curve: Review Market Concentration")
axes[0].set_xlabel("Cumulative Share of Games")
axes[0].set_ylabel("Cumulative Share of Reviews")
axes[0].legend()

# Plot 2: Correlation Heatmap
corr_cols = {
    "Log Reviews": np.log1p(mature["total_reviews"]),
    "Price": mature["price"],
    "Languages": mature["n_languages"],
    "Screenshots": mature["n_screenshots"],
    "Desc Length": mature["desc_len"],
    "Platforms": mature["n_platforms"],
}
corr = pd.DataFrame(corr_cols).corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[1])
axes[1].set_title("Spearman Rank Correlations (Launch Features)")

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.savefig("steam_eda_dashboard.png", dpi=300, bbox_inches="tight")
print("Saved dashboard figure to steam_eda_dashboard.png")
plt.show()