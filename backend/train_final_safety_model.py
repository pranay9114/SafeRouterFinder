# ============================================================
# SAFE ROUTE FINDER
# FINAL MULTI-BRANCH DEEP LEARNING SAFETY MODEL
#
# Inputs:
#   Traffic
#   Lighting
#   Weather
#   Crime
#   Emergency Access
#
# Architecture:
#   5 Deep Learning Branches
#          ↓
#   5 Feature Tokens
#          ↓
#   Multi-Head Attention
#          ↓
#   Transformer-style Residual Block
#          ↓
#   Feature Fusion
#          ↓
#   Deep Neural Network
#          ↓
#   Safety Score 0-100
#
# Target:
#   safety_score
#
# IMPORTANT:
#   risk_level is NOT used as an input.
#   route_id is used ONLY for route-level splitting.
# ============================================================

import os
import random
import warnings
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Concatenate,
    Add,
    Reshape,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 75)
print("SAFE ROUTE FINDER - FINAL DEEP LEARNING MODEL")
print("=" * 75)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "training",
    "route_safety_dataset_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "safe_route_final_multibranch.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "safe_route_final_scalers.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "safe_route_final_features.txt"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "safe_route_final_metrics.txt"
)


# ============================================================
# 3. CUSTOM SERIALIZABLE SAFETY SCORE LAYER
# ============================================================
#
# This replaces the previous Lambda layer.
#
# Keras can safely save and reload this layer.
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="SafeRoute"
)
class ScaleTo100(tf.keras.layers.Layer):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

    def call(
        self,
        inputs
    ):
        return inputs * 100.0

    def get_config(
        self
    ):
        config = super().get_config()
        return config


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(
    DATASET_PATH
):
    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATASET_PATH
    )

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset loaded successfully."
)

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "route_id",
    "district",
    "date",
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
    "safety_score",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print(
        "\nMissing columns:"
    )

    for column in missing_columns:
        print(
            " -",
            column
        )

    raise ValueError(
        "Required columns are missing."
    )

print(
    "\nAll required columns are present."
)


# ============================================================
# 6. CLEAN DATA
# ============================================================

print(
    "\n[2] Cleaning dataset..."
)

df = df.copy()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_columns = [
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
    "safety_score",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

before_rows = len(df)

df = df.dropna(
    subset=required_columns
).reset_index(
    drop=True
)

after_rows = len(df)

print(
    "Rows before cleaning:",
    before_rows
)

print(
    "Rows after cleaning :",
    after_rows
)


# ============================================================
# 7. SAFETY TARGET VALIDATION
# ============================================================

print(
    "\n[3] Safety target validation..."
)

df["safety_score"] = (
    df["safety_score"]
    .clip(0, 100)
)

print(
    "Minimum safety score:",
    df["safety_score"].min()
)

print(
    "Maximum safety score:",
    df["safety_score"].max()
)

print(
    "Mean safety score   :",
    round(
        df["safety_score"].mean(),
        4
    )
)

print(
    "Safety score std    :",
    round(
        df["safety_score"].std(),
        4
    )
)


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print(
    "\n[4] Creating engineered features..."
)


# ------------------------------------------------------------
# Traffic
# ------------------------------------------------------------

df["traffic_safety_score"] = (
    df["traffic_index"]
).clip(
    0,
    100
)


# ------------------------------------------------------------
# Rainfall risk
# ------------------------------------------------------------

rainfall_max = max(
    df["rainfall_mm"].max(),
    1.0
)

df["rainfall_risk"] = (
    df["rainfall_mm"]
    / rainfall_max
    * 100
).clip(
    0,
    100
)


# ------------------------------------------------------------
# Visibility safety
# ------------------------------------------------------------

visibility_min = (
    df["visibility_km"].min()
)

visibility_max = (
    df["visibility_km"].max()
)

if visibility_max > visibility_min:

    df["visibility_safety_score"] = (
        (
            df["visibility_km"]
            - visibility_min
        )
        /
        (
            visibility_max
            - visibility_min
        )
        * 100
    )

else:

    df["visibility_safety_score"] = 100.0


df["visibility_safety_score"] = (
    df["visibility_safety_score"]
    .clip(
        0,
        100
    )
)


# ------------------------------------------------------------
# Crime intensity
# ------------------------------------------------------------

crime_median = max(
    df["total_crime_count"].median(),
    1.0
)

df["crime_intensity"] = (
    df["total_crime_count"]
    / crime_median
)


# ------------------------------------------------------------
# Crime pressure
# ------------------------------------------------------------

df["crime_pressure"] = (

    0.25
    * df["violent_crime_count"]

    +

    0.20
    * df["property_crime_count"]

    +

    0.20
    * df["women_crime_count"]

    +

    0.15
    * df["other_crime_count"]

    +

    0.20
    * df["crime_rate_per_1000_exposure"]
)


# ------------------------------------------------------------
# Emergency distance
# ------------------------------------------------------------

df["emergency_distance"] = (
    df["nearest_police_km"]
    +
    df["nearest_hospital_km"]
)


# ------------------------------------------------------------
# Monthly seasonal features
# ------------------------------------------------------------

df["month"] = (
    df["date"]
    .dt
    .month
)

df["month_sin"] = np.sin(
    2
    * np.pi
    * df["month"]
    / 12
)

df["month_cos"] = np.cos(
    2
    * np.pi
    * df["month"]
    / 12
)


print(
    "Feature engineering complete."
)


# ============================================================
# 9. FIVE FEATURE BRANCHES
# ============================================================

traffic_features = [
    "traffic_index",
    "traffic_safety_score",
]

lighting_features = [
    "lighting_score",
]

weather_features = [
    "rainfall_mm",
    "rainfall_risk",
    "visibility_km",
    "visibility_safety_score",
    "month_sin",
    "month_cos",
]

crime_features = [
    "violent_crime_count",
    "property_crime_count",
    "women_crime_count",
    "other_crime_count",
    "total_crime_count",
    "crime_rate_per_1000_exposure",
    "crime_intensity",
    "crime_pressure",
]

emergency_features = [
    "nearest_police_km",
    "nearest_hospital_km",
    "emergency_access_score",
    "emergency_distance",
]


all_features = (
    traffic_features
    +
    lighting_features
    +
    weather_features
    +
    crime_features
    +
    emergency_features
)


print(
    "\nFeature groups:"
)

print(
    "Traffic   :",
    traffic_features
)

print(
    "Lighting  :",
    lighting_features
)

print(
    "Weather   :",
    weather_features
)

print(
    "Crime     :",
    crime_features
)

print(
    "Emergency :",
    emergency_features
)

print(
    "\nTotal model input features:",
    len(all_features)
)


# ============================================================
# 10. TARGET AND GROUPS
# ============================================================

TARGET = "safety_score"

X = df[
    all_features
].copy()

y = df[
    TARGET
].astype(
    np.float32
).values

groups = df[
    "route_id"
].astype(
    str
).values


# ============================================================
# 11. ROUTE-LEVEL TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[5] Creating route-level train/test split..."
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[
    train_indices
].copy()

X_test = X.iloc[
    test_indices
].copy()

y_train = y[
    train_indices
]

y_test = y[
    test_indices
]


train_route_count = (
    df.iloc[
        train_indices
    ]["route_id"]
    .nunique()
)

test_route_count = (
    df.iloc[
        test_indices
    ]["route_id"]
    .nunique()
)


print(
    "Training rows :",
    len(X_train)
)

print(
    "Testing rows  :",
    len(X_test)
)

print(
    "Training routes:",
    train_route_count
)

print(
    "Testing routes :",
    test_route_count
)


# ============================================================
# 12. SCALE EACH BRANCH
# ============================================================

print(
    "\n[6] Scaling feature branches..."
)

scalers = {}


def scale_branch(
    train_dataframe,
    test_dataframe,
    feature_list,
    branch_name
):

    scaler = StandardScaler()

    train_values = scaler.fit_transform(
        train_dataframe[
            feature_list
        ]
    )

    test_values = scaler.transform(
        test_dataframe[
            feature_list
        ]
    )

    scalers[
        branch_name
    ] = scaler

    return (
        train_values.astype(
            np.float32
        ),
        test_values.astype(
            np.float32
        )
    )


X_train_traffic, X_test_traffic = scale_branch(
    X_train,
    X_test,
    traffic_features,
    "traffic"
)


X_train_lighting, X_test_lighting = scale_branch(
    X_train,
    X_test,
    lighting_features,
    "lighting"
)


X_train_weather, X_test_weather = scale_branch(
    X_train,
    X_test,
    weather_features,
    "weather"
)


X_train_crime, X_test_crime = scale_branch(
    X_train,
    X_test,
    crime_features,
    "crime"
)


X_train_emergency, X_test_emergency = scale_branch(
    X_train,
    X_test,
    emergency_features,
    "emergency"
)


print(
    "Scaling complete."
)


# ============================================================
# 13. BRANCH NETWORK
# ============================================================

def create_branch(
    input_tensor,
    first_units=64
):

    x = Dense(
        first_units,
        activation="relu"
    )(input_tensor)

    x = BatchNormalization()(x)

    x = Dense(
        first_units,
        activation="relu"
    )(x)

    x = Dropout(
        0.15
    )(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    return x


# ============================================================
# 14. INPUT LAYERS
# ============================================================

traffic_input = Input(
    shape=(
        len(traffic_features),
    ),
    name="traffic_input"
)

lighting_input = Input(
    shape=(
        len(lighting_features),
    ),
    name="lighting_input"
)

weather_input = Input(
    shape=(
        len(weather_features),
    ),
    name="weather_input"
)

crime_input = Input(
    shape=(
        len(crime_features),
    ),
    name="crime_input"
)

emergency_input = Input(
    shape=(
        len(emergency_features),
    ),
    name="emergency_input"
)


# ============================================================
# 15. FIVE BRANCHES
# ============================================================

traffic_branch = create_branch(
    traffic_input,
    64
)

lighting_branch = create_branch(
    lighting_input,
    64
)

weather_branch = create_branch(
    weather_input,
    64
)

crime_branch = create_branch(
    crime_input,
    96
)

emergency_branch = create_branch(
    emergency_input,
    64
)


# ============================================================
# 16. FIVE FEATURE TOKENS
# ============================================================

traffic_token = Reshape(
    (1, 32),
    name="traffic_token"
)(
    traffic_branch
)

lighting_token = Reshape(
    (1, 32),
    name="lighting_token"
)(
    lighting_branch
)

weather_token = Reshape(
    (1, 32),
    name="weather_token"
)(
    weather_branch
)

crime_token = Reshape(
    (1, 32),
    name="crime_token"
)(
    crime_branch
)

emergency_token = Reshape(
    (1, 32),
    name="emergency_token"
)(
    emergency_branch
)


tokens = Concatenate(
    axis=1,
    name="five_feature_tokens"
)(
    [
        traffic_token,
        lighting_token,
        weather_token,
        crime_token,
        emergency_token,
    ]
)


# ============================================================
# 17. MULTI-HEAD ATTENTION
# ============================================================

attention_output = MultiHeadAttention(
    num_heads=4,
    key_dim=8,
    dropout=0.10,
    name="multi_head_attention"
)(
    query=tokens,
    value=tokens,
    key=tokens
)


# ============================================================
# 18. ATTENTION RESIDUAL
# ============================================================

attention_residual = Add(
    name="attention_residual"
)(
    [
        tokens,
        attention_output
    ]
)


attention_norm = LayerNormalization(
    name="attention_layer_norm"
)(
    attention_residual
)


# ============================================================
# 19. FEED-FORWARD TRANSFORMER BLOCK
# ============================================================

ffn = Dense(
    64,
    activation="relu",
    name="feed_forward_1"
)(
    attention_norm
)

ffn = Dropout(
    0.15,
    name="feed_forward_dropout"
)(
    ffn
)

ffn = Dense(
    32,
    activation=None,
    name="feed_forward_2"
)(
    ffn
)


transformer_residual = Add(
    name="transformer_residual"
)(
    [
        attention_norm,
        ffn
    ]
)


transformer_output = LayerNormalization(
    name="transformer_layer_norm"
)(
    transformer_residual
)


# ============================================================
# 20. FEATURE FUSION
# ============================================================

fused_features = GlobalAveragePooling1D(
    name="feature_fusion"
)(
    transformer_output
)


# ============================================================
# 21. FINAL DEEP NETWORK
# ============================================================

x = Dense(
    128,
    activation="relu",
    name="fusion_dense_128"
)(
    fused_features
)

x = BatchNormalization(
    name="fusion_batch_norm_128"
)(
    x
)

x = Dropout(
    0.20,
    name="fusion_dropout_128"
)(
    x
)

x = Dense(
    64,
    activation="relu",
    name="fusion_dense_64"
)(
    x
)

x = BatchNormalization(
    name="fusion_batch_norm_64"
)(
    x
)

x = Dropout(
    0.15,
    name="fusion_dropout_64"
)(
    x
)

x = Dense(
    32,
    activation="relu",
    name="fusion_dense_32"
)(
    x
)

x = Dense(
    16,
    activation="relu",
    name="fusion_dense_16"
)(
    x
)


# ============================================================
# 22. SAFETY SCORE OUTPUT
# ============================================================

safety_probability = Dense(
    1,
    activation="sigmoid",
    name="safety_probability"
)(
    x
)

safety_output = ScaleTo100(
    name="safety_score"
)(
    safety_probability
)


# ============================================================
# 23. CREATE MODEL
# ============================================================

model = Model(
    inputs=[
        traffic_input,
        lighting_input,
        weather_input,
        crime_input,
        emergency_input,
    ],
    outputs=safety_output,
    name="SafeRoute_Final_MultiBranch_Attention"
)


# ============================================================
# 24. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        ),
        tf.keras.metrics.RootMeanSquaredError(
            name="rmse"
        ),
    ]
)


# ============================================================
# 25. MODEL SUMMARY
# ============================================================

print(
    "\n[7] MODEL ARCHITECTURE"
)

print(
    "-" * 75
)

model.summary()


# ============================================================
# 26. CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]


# ============================================================
# 27. TRAINING
# ============================================================

print(
    "\n[8] STARTING TRAINING"
)

print(
    "-" * 75
)

history = model.fit(

    [
        X_train_traffic,
        X_train_lighting,
        X_train_weather,
        X_train_crime,
        X_train_emergency,
    ],

    y_train,

    validation_split=0.15,

    epochs=250,

    batch_size=64,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 28. BEST MODEL ALREADY RESTORED
# ============================================================
#
# Because EarlyStopping uses:
#   restore_best_weights=True
#
# and ModelCheckpoint saved the best model,
# we do NOT need Lambda deserialization.
#
# We still reload it to verify that the saved .keras file
# can be opened correctly.
# ============================================================

print(
    "\n[9] Verifying saved model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Saved model loaded successfully."
)


# ============================================================
# 29. TEST PREDICTIONS
# ============================================================

print(
    "\n[10] Generating test predictions..."
)

predictions = model.predict(

    [
        X_test_traffic,
        X_test_lighting,
        X_test_weather,
        X_test_crime,
        X_test_emergency,
    ],

    verbose=0
).reshape(-1)


predictions = np.clip(
    predictions,
    0,
    100
)


# ============================================================
# 30. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ------------------------------------------------------------
# Score closeness
# ------------------------------------------------------------
#
# This is a project metric indicating how close predictions
# are to the actual safety scores.
#
# It is NOT "real-world accuracy".
# ------------------------------------------------------------

mean_actual = np.mean(
    np.abs(y_test)
)

if mean_actual > 0:

    score_closeness = max(
        0,
        100
        *
        (
            1
            -
            (
                np.mean(
                    np.abs(
                        y_test
                        -
                        predictions
                    )
                )
                /
                mean_actual
            )
        )
    )

else:

    score_closeness = 0


# ============================================================
# 31. RISK CLASSIFICATION
# ============================================================

def safety_to_class(
    score
):

    if score < 30:
        return 0

    elif score < 45:
        return 1

    elif score < 60:
        return 2

    elif score < 75:
        return 3

    else:
        return 4


actual_classes = np.array(
    [
        safety_to_class(
            value
        )
        for value in y_test
    ]
)


predicted_classes = np.array(
    [
        safety_to_class(
            value
        )
        for value in predictions
    ]
)


classification_accuracy = (
    accuracy_score(
        actual_classes,
        predicted_classes
    )
)


# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

class_names = [
    "Very High Risk",
    "High Risk",
    "Medium Risk",
    "Low Risk",
    "Very Low Risk",
]


report = classification_report(

    actual_classes,

    predicted_classes,

    labels=[
        0,
        1,
        2,
        3,
        4,
    ],

    target_names=class_names,

    zero_division=0
)


confusion = confusion_matrix(

    actual_classes,

    predicted_classes,

    labels=[
        0,
        1,
        2,
        3,
        4,
    ]
)


# ============================================================
# 33. PRINT RESULTS
# ============================================================

print(
    "\n"
)

print(
    "=" * 75
)

print(
    "FINAL MODEL RESULTS"
)

print(
    "=" * 75
)

print(
    f"\nMAE                     : {mae:.4f}"
)

print(
    f"RMSE                    : {rmse:.4f}"
)

print(
    f"R²                      : {r2:.4f}"
)

print(
    f"Score closeness         : "
    f"{score_closeness:.2f}%"
)

print(
    f"5-Class Risk Accuracy   : "
    f"{classification_accuracy * 100:.2f}%"
)


print(
    "\nRisk classification report:"
)

print(
    report
)


print(
    "Confusion Matrix:"
)

print(
    confusion
)


# ============================================================
# 34. SAMPLE PREDICTIONS
# ============================================================

print(
    "\nSample predictions:"
)

print(
    "-" * 75
)


sample_count = min(
    20,
    len(y_test)
)


for i in range(
    sample_count
):

    actual = y_test[i]

    predicted = predictions[i]

    difference = abs(
        actual
        -
        predicted
    )

    print(
        f"{i + 1:02d}. "
        f"Actual: {actual:6.2f}   "
        f"Predicted: {predicted:6.2f}   "
        f"Difference: {difference:6.2f}"
    )


# ============================================================
# 35. SAVE SCALERS
# ============================================================

print(
    "\n[11] Saving scalers..."
)

joblib.dump(

    {
        "traffic": scalers[
            "traffic"
        ],

        "lighting": scalers[
            "lighting"
        ],

        "weather": scalers[
            "weather"
        ],

        "crime": scalers[
            "crime"
        ],

        "emergency": scalers[
            "emergency"
        ],
    },

    SCALER_PATH
)


# ============================================================
# 36. SAVE FEATURE CONFIGURATION
# ============================================================

print(
    "Saving feature configuration..."
)

with open(
    FEATURE_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SAFE ROUTE FINDER FINAL MODEL FEATURES\n"
    )

    file.write(
        "=" * 60
        +
        "\n"
    )

    file.write(
        "\nTraffic Features:\n"
    )

    for feature in traffic_features:

        file.write(
            feature
            +
            "\n"
        )


    file.write(
        "\nLighting Features:\n"
    )

    for feature in lighting_features:

        file.write(
            feature
            +
            "\n"
        )


    file.write(
        "\nWeather Features:\n"
    )

    for feature in weather_features:

        file.write(
            feature
            +
            "\n"
        )


    file.write(
        "\nCrime Features:\n"
    )

    for feature in crime_features:

        file.write(
            feature
            +
            "\n"
        )


    file.write(
        "\nEmergency Features:\n"
    )

    for feature in emergency_features:

        file.write(
            feature
            +
            "\n"
        )


    file.write(
        "\nTarget:\n"
    )

    file.write(
        "safety_score\n"
    )


    file.write(
        "\nExcluded columns:\n"
    )

    file.write(
        "risk_level\n"
    )

    file.write(
        "route_id\n"
    )

    file.write(
        "route_name\n"
    )


# ============================================================
# 37. SAVE METRICS
# ============================================================

print(
    "Saving metrics..."
)

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SAFE ROUTE FINDER - FINAL MODEL RESULTS\n"
    )

    file.write(
        "=" * 60
        +
        "\n"
    )

    file.write(
        f"Dataset rows: {len(df)}\n"
    )

    file.write(
        f"Training rows: {len(X_train)}\n"
    )

    file.write(
        f"Testing rows: {len(X_test)}\n"
    )

    file.write(
        f"Training routes: {train_route_count}\n"
    )

    file.write(
        f"Testing routes: {test_route_count}\n"
    )

    file.write(
        f"\nMAE: {mae:.6f}\n"
    )

    file.write(
        f"RMSE: {rmse:.6f}\n"
    )

    file.write(
        f"R2: {r2:.6f}\n"
    )

    file.write(
        f"Score closeness: "
        f"{score_closeness:.2f}%\n"
    )

    file.write(
        f"5-Class Risk Accuracy: "
        f"{classification_accuracy * 100:.2f}%\n"
    )


# ============================================================
# 38. FINAL OUTPUT
# ============================================================

print(
    "\n"
)

print(
    "=" * 75
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 75
)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nScalers:"
)

print(
    SCALER_PATH
)

print(
    "\nFeatures:"
)

print(
    FEATURE_PATH
)

print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)

print(
    "\nIMPORTANT:"
)

print(
    "Score closeness is NOT real-world accuracy."
)

print(
    "5-Class Risk Accuracy is classification accuracy "
    "on the held-out test routes."
)

print(
    "\nThe saved Keras model was successfully reloaded."
)

print(
    "\nNext stage:"
)

print(
    "Connect the trained model to api.py for live route prediction."
)

print(
    "=" * 75
)