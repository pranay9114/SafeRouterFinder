import pandas as pd
import os

# ============================================================
# SAFE ROUTE FINDER - DATASET INSPECTION
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "datasets",
    "training",
    "route_safety_dataset_cleaned.csv"
)

print("=" * 70)
print("SAFE ROUTE FINDER - TRAINING DATASET INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Check file
# ------------------------------------------------------------

print("\n[1] Checking dataset file...")

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset not found!")
    print("Expected path:")
    print(DATASET_PATH)
    raise SystemExit(1)

print("Dataset found:")
print(DATASET_PATH)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\n[2] Dataset shape")
print("-" * 70)
print("Rows    :", len(df))
print("Columns :", len(df.columns))

# ------------------------------------------------------------
# 3. Column names
# ------------------------------------------------------------

print("\n[3] Columns")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# ------------------------------------------------------------
# 4. First 5 rows
# ------------------------------------------------------------

print("\n[4] First 5 records")
print("-" * 70)

print(df.head().to_string())

# ------------------------------------------------------------
# 5. Missing values
# ------------------------------------------------------------

print("\n[5] Missing values")
print("-" * 70)

missing = df.isnull().sum()

missing_found = False

for column, count in missing.items():
    if count > 0:
        missing_found = True
        print(f"{column}: {count}")

if not missing_found:
    print("No missing values found.")

# ------------------------------------------------------------
# 6. Duplicate rows
# ------------------------------------------------------------

print("\n[6] Duplicate records")
print("-" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# ------------------------------------------------------------
# 7. Safety score analysis
# ------------------------------------------------------------

print("\n[7] SAFETY SCORE ANALYSIS")
print("-" * 70)

if "safety_score" not in df.columns:
    print("ERROR: safety_score column not found!")
    raise SystemExit(1)

print("Minimum :", df["safety_score"].min())
print("Maximum :", df["safety_score"].max())
print("Mean    :", df["safety_score"].mean())
print("Median  :", df["safety_score"].median())
print("Std Dev :", df["safety_score"].std())

# ------------------------------------------------------------
# 8. Safety score distribution
# ------------------------------------------------------------

print("\n[8] Safety score distribution")
print("-" * 70)

bins = [
    -float("inf"),
    30,
    45,
    60,
    75,
    float("inf")
]

labels = [
    "Very High Risk (<30)",
    "High Risk (30-45)",
    "Medium Risk (45-60)",
    "Low Risk (60-75)",
    "Very Low Risk (>75)"
]

distribution = pd.cut(
    df["safety_score"],
    bins=bins,
    labels=labels
).value_counts().sort_index()

for label, count in distribution.items():
    percentage = count / len(df) * 100
    print(f"{label:25} : {count:5} ({percentage:6.2f}%)")

# ------------------------------------------------------------
# 9. Numeric feature statistics
# ------------------------------------------------------------

print("\n[9] Numeric feature statistics")
print("-" * 70)

numeric_columns = df.select_dtypes(include="number").columns

print(
    df[numeric_columns]
    .describe()
    .round(2)
    .to_string()
)

# ------------------------------------------------------------
# 10. Check target leakage columns
# ------------------------------------------------------------

print("\n[10] TARGET LEAKAGE CHECK")
print("-" * 70)

possible_leakage = [
    "risk_level",
    "safety_score",
    "risk_score",
    "safety_label",
    "target"
]

for column in possible_leakage:
    if column in df.columns:
        print(f"WARNING: {column} exists in dataset")

print("\nFor final model:")
print("safety_score = TARGET")
print("risk_level   = DO NOT USE AS INPUT")

# ------------------------------------------------------------
# 11. Correlation with safety score
# ------------------------------------------------------------

print("\n[11] Correlation with safety_score")
print("-" * 70)

correlations = (
    df[numeric_columns]
    .corr()["safety_score"]
    .drop("safety_score")
    .sort_values(ascending=False)
)

print(correlations.round(3).to_string())

# ------------------------------------------------------------
# 12. Route information
# ------------------------------------------------------------

print("\n[12] ROUTE INFORMATION")
print("-" * 70)

if "route_id" in df.columns:
    print("Unique routes:", df["route_id"].nunique())

if "route_name" in df.columns:
    print("Unique route names:", df["route_name"].nunique())

if "district" in df.columns:
    print("Unique districts:", df["district"].nunique())

if "date" in df.columns:
    print("Date range:")
    print("Minimum:", df["date"].min())
    print("Maximum:", df["date"].max())

# ------------------------------------------------------------
# 13. Crime features
# ------------------------------------------------------------

print("\n[13] CRIME FEATURES")
print("-" * 70)

crime_columns = [
    "violent_crime_count",
    "property_crime_count",
    "women_crime_count",
    "other_crime_count",
    "total_crime_count",
    "crime_rate_per_1000_exposure"
]

for column in crime_columns:
    if column in df.columns:
        print(
            f"{column:35} "
            f"min={df[column].min():.2f}, "
            f"max={df[column].max():.2f}, "
            f"mean={df[column].mean():.2f}"
        )

# ------------------------------------------------------------
# 14. Environment features
# ------------------------------------------------------------

print("\n[14] ENVIRONMENT FEATURES")
print("-" * 70)

environment_columns = [
    "lighting_score",
    "rainfall_mm",
    "visibility_km",
    "traffic_index"
]

for column in environment_columns:
    if column in df.columns:
        print(
            f"{column:25} "
            f"min={df[column].min():.2f}, "
            f"max={df[column].max():.2f}, "
            f"mean={df[column].mean():.2f}"
        )

# ------------------------------------------------------------
# 15. Emergency access features
# ------------------------------------------------------------

print("\n[15] EMERGENCY ACCESS FEATURES")
print("-" * 70)

emergency_columns = [
    "nearest_police_km",
    "nearest_hospital_km",
    "emergency_access_score"
]

for column in emergency_columns:
    if column in df.columns:
        print(
            f"{column:30} "
            f"min={df[column].min():.2f}, "
            f"max={df[column].max():.2f}, "
            f"mean={df[column].mean():.2f}"
        )

# ------------------------------------------------------------
# 16. Final validation
# ------------------------------------------------------------

print("\n[16] FINAL VALIDATION")
print("-" * 70)

required_columns = [
    "route_id",
    "district",
    "traffic_index",
    "lighting_score",
    "rainfall_mm",
    "visibility_km",
    "violent_crime_count",
    "property_crime_count",
    "women_crime_count",
    "other_crime_count",
    "total_crime_count",
    "crime_rate_per_1000_exposure",
    "nearest_police_km",
    "nearest_hospital_km",
    "emergency_access_score",
    "safety_score"
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    print("Missing required columns:")
    for column in missing_required:
        print(" -", column)
else:
    print("All required ML columns are present.")

# ------------------------------------------------------------
# 17. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print("\nNext step:")
print("We will use these results to build the FINAL multi-branch")
print("deep-learning safety model with feature fusion + attention.")

print("\nIMPORTANT:")
print("Do NOT train the model yet.")
print("Send me the complete output from this script first.")