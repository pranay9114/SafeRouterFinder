import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "datasets/training/route_training_features.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "safe_route_advanced_dnn.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "safe_route_advanced_scaler.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "advanced_model_features.txt"
)

RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FEATURE GROUPS
# ============================================================

LIGHTING_FEATURES = [
    "lighting_score",
    "lighting_value",
]

TRAFFIC_FEATURES = [
    "traffic_score",
    "average_current_speed_kmph",
    "average_free_flow_speed_kmph",
    "average_delay_seconds",
    "average_speed_ratio",
    "traffic_confidence",
]

WEATHER_FEATURES = [
    "weather_score",
    "average_temperature_c",
    "average_humidity_percent",
    "average_precipitation_mm",
    "average_rain_mm",
    "average_wind_speed_kmph",
    "average_wind_gusts_kmph",
    "average_cloud_cover_percent",
]

CRIME_FEATURES = [
    "crime_risk_score",
    "crime_safety_score",
    "violent_crime_cases",
    "property_crime_cases",
    "women_safety_crime_cases",
    "total_crime_cases",
]

ROUTE_FEATURES = [
    "distance_km",
    "time_minutes",
]


ALL_FEATURES = (
    LIGHTING_FEATURES
    + TRAFFIC_FEATURES
    + WEATHER_FEATURES
    + CRIME_FEATURES
    + ROUTE_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFE ROUTE FINDER")
print("ADVANCED MULTI-BRANCH DEEP LEARNING MODEL")
print("=" * 70)

print("\nLoading training data...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nTraining dataset not found:\n{DATA_FILE}\n"
        "Run training_data_generator.py first."
    )

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in ALL_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "\nMissing required features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

for feature in ALL_FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

df[ALL_FEATURES] = df[ALL_FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

df[ALL_FEATURES] = df[ALL_FEATURES].fillna(
    df[ALL_FEATURES].median()
)


# ============================================================
# PROVISIONAL PROTOTYPE TARGET
# ============================================================
#
# IMPORTANT:
#
# This is NOT a real-world safety label.
#
# It is only used to verify that the complete deep-learning
# pipeline works.
#
# For the final capstone model we MUST replace this with
# genuine route-level safety labels.
#
# ============================================================

print("\nCreating provisional prototype target...")

traffic_delay_safety = (
    100
    - (
        df["average_delay_seconds"]
        .clip(lower=0, upper=300)
        / 300
        * 100
    )
)

df["prototype_safety_target"] = (
    0.25 * df["lighting_score"].clip(0, 100)
    + 0.20 * df["traffic_score"].clip(0, 100)
    + 0.20 * df["weather_score"].clip(0, 100)
    + 0.30 * df["crime_safety_score"].clip(0, 100)
    + 0.05 * traffic_delay_safety
).clip(0, 100)


y = df["prototype_safety_target"].values.astype(
    np.float32
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

X = df[ALL_FEATURES].values.astype(
    np.float32
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")


# ============================================================
# SPLIT FEATURES INTO BRANCHES
# ============================================================

lighting_count = len(LIGHTING_FEATURES)
traffic_count = len(TRAFFIC_FEATURES)
weather_count = len(WEATHER_FEATURES)
crime_count = len(CRIME_FEATURES)
route_count = len(ROUTE_FEATURES)


def split_features(X):

    start = 0

    lighting = X[
        :,
        start:start + lighting_count
    ]

    start += lighting_count

    traffic = X[
        :,
        start:start + traffic_count
    ]

    start += traffic_count

    weather = X[
        :,
        start:start + weather_count
    ]

    start += weather_count

    crime = X[
        :,
        start:start + crime_count
    ]

    start += crime_count

    route = X[
        :,
        start:start + route_count
    ]

    return (
        lighting,
        traffic,
        weather,
        crime,
        route
    )


(
    lighting_train,
    traffic_train,
    weather_train,
    crime_train,
    route_train
) = split_features(X_train)


(
    lighting_test,
    traffic_test,
    weather_test,
    crime_test,
    route_test
) = split_features(X_test)


# ============================================================
# BRANCH CREATOR
# ============================================================

def create_branch(
    input_tensor,
    units1,
    units2
):

    x = Dense(
        units1,
        activation="relu"
    )(input_tensor)

    x = BatchNormalization()(x)

    x = Dropout(0.20)(x)

    x = Dense(
        units2,
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(0.15)(x)

    return x


# ============================================================
# INPUTS
# ============================================================

lighting_input = Input(
    shape=(lighting_count,),
    name="lighting_input"
)

traffic_input = Input(
    shape=(traffic_count,),
    name="traffic_input"
)

weather_input = Input(
    shape=(weather_count,),
    name="weather_input"
)

crime_input = Input(
    shape=(crime_count,),
    name="crime_input"
)

route_input = Input(
    shape=(route_count,),
    name="route_input"
)


# ============================================================
# INDIVIDUAL NEURAL BRANCHES
# ============================================================

lighting_branch = create_branch(
    lighting_input,
    32,
    16
)

traffic_branch = create_branch(
    traffic_input,
    48,
    24
)

weather_branch = create_branch(
    weather_input,
    48,
    24
)

crime_branch = create_branch(
    crime_input,
    48,
    24
)

route_branch = create_branch(
    route_input,
    32,
    16
)


# ============================================================
# FEATURE FUSION
# ============================================================

fusion = Concatenate(
    name="feature_fusion"
)([
    lighting_branch,
    traffic_branch,
    weather_branch,
    crime_branch,
    route_branch
])


# ============================================================
# FUSION NETWORK
# ============================================================

fusion = Dense(
    128,
    activation="relu",
    name="fusion_dense_128"
)(fusion)

fusion = BatchNormalization()(fusion)

fusion = Dropout(0.25)(fusion)

fusion = Dense(
    64,
    activation="relu",
    name="fusion_dense_64"
)(fusion)

fusion = BatchNormalization()(fusion)


# ============================================================
# ATTENTION
# ============================================================
#
# Convert the fused feature vector into a sequence of length 1
# so MultiHeadAttention can learn feature interactions.
#
# ============================================================

attention_input = Reshape(
    (1, 64),
    name="attention_reshape"
)(fusion)


attention_output = MultiHeadAttention(
    num_heads=4,
    key_dim=16,
    dropout=0.10,
    name="feature_attention"
)(
    attention_input,
    attention_input
)


attention_output = LayerNormalization()(
    attention_output
)


attention_output = GlobalAveragePooling1D()(
    attention_output
)


# ============================================================
# FINAL DEEP NETWORK
# ============================================================

x = Dense(
    64,
    activation="relu",
    name="final_dense_64"
)(attention_output)

x = BatchNormalization()(x)

x = Dropout(0.20)(x)

x = Dense(
    32,
    activation="relu",
    name="final_dense_32"
)(x)

x = Dropout(0.15)(x)

x = Dense(
    16,
    activation="relu",
    name="final_dense_16"
)(x)


# ============================================================
# OUTPUT
# ============================================================

output = Dense(
    1,
    activation="sigmoid",
    name="safety_output"
)(x)

# Convert 0-1 output into 0-100 safety score
output = tf.keras.layers.Lambda(
    lambda value: value * 100,
    name="safety_score"
)(output)


# ============================================================
# MODEL
# ============================================================

model = Model(
    inputs=[
        lighting_input,
        traffic_input,
        weather_input,
        crime_input,
        route_input
    ],
    outputs=output
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=[
        "mae"
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
]


history = model.fit(

    [
        lighting_train,
        traffic_train,
        weather_train,
        crime_train,
        route_train
    ],

    y_train,

    validation_split=0.20,

    epochs=200,

    batch_size=16,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

predictions = model.predict(
    [
        lighting_test,
        traffic_test,
        weather_test,
        crime_test,
        route_test
    ],
    verbose=0
).flatten()


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


print()
print("=" * 70)
print("PROTOTYPE MODEL RESULTS")
print("=" * 70)

print(f"MAE :  {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²  :  {r2:.4f}")

print("=" * 70)

print()
print("IMPORTANT:")
print(
    "These metrics are NOT real-world safety accuracy."
)

print(
    "The current target is a provisional prototype target."
)

print(
    "A genuine safety label is required for final evaluation."
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)

print(
    f"\nModel saved to:\n{MODEL_FILE}"
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    f"Scaler saved to:\n{SCALER_FILE}"
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as file:

    for feature in ALL_FEATURES:

        file.write(
            feature + "\n"
        )


print(
    f"Feature list saved to:\n{FEATURE_FILE}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

for actual, predicted in zip(
    y_test[:10],
    predictions[:10]
):

    print(
        f"Actual: {actual:.2f} "
        f"| Predicted: {predicted:.2f}"
    )


print()
print("=" * 70)
print("ADVANCED MODEL TRAINING COMPLETE")
print("=" * 70)