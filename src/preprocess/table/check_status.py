import pandas as pd
import numpy as np
from pathlib import Path

def _get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


REPO_ROOT = _get_repo_root()
csv_path = REPO_ROOT / "data" / "metadata" / "now.csv"
df = pd.read_csv(csv_path)
# 显示所有行
pd.set_option('display.max_rows', None)

# ===== 1. Drop unnecessary columns =====
drop_cols = [
    "content_status", "download_status", "backers_count", "percent_funded", "usd_pledged",
    "project_id", "creator_id", "project_url", "cover_url", "creator_profile_url",
    "title", "blurb", "staff_pick"
]
df.drop(columns=drop_cols, inplace=True)

# ===== 2. Label creation =====
y = (df["state"] == "successful").astype(int)
X = df.drop(columns=["state"])

# ===== 3. Create 'duration_days' from 'launched_at' and 'deadline' =====
X["launched_at"] = pd.to_datetime(X["launched_at"], utc=True)
X["deadline"] = pd.to_datetime(X["deadline"], utc=True)
X["duration_days"] = (X["deadline"] - X["launched_at"]).dt.total_seconds() / 86400.0
X.drop(columns=["launched_at", "deadline"], inplace=True)

# ===== 4. Apply log transformation to 'usd_goal' =====
X["log_usd_goal"] = np.log1p(X["usd_goal"])
X.drop(columns=["usd_goal"], inplace=True)

# ===== 5. Quick sanity check =====
print(f"X shape: {X.shape}")
print(f"y mean (success rate): {y.mean():.4f}")

# ===== 6. Data overview =====
# Numeric columns
print("\nNumeric columns:")
print(X.select_dtypes(include=[np.number]).columns.tolist())

# Categorical columns
print("\nCategorical columns:")
print(X.select_dtypes(include="object").columns.tolist())

# Missing data percentage
print("\nMissing rate (%):")
missing_rate = X.isna().mean() * 100
print(missing_rate.sort_values(ascending=False).head(10))

# ===== 7. Basic statistics of numeric columns =====
print("\nBasic statistics of numeric columns:")
print(X.describe())

# ===== 8. Value counts for categorical columns =====
print("\nValue counts of categorical columns:")
for col in X.select_dtypes(include="object").columns:
    print(f"\n{col} value counts:")
    print(X[col].value_counts())
