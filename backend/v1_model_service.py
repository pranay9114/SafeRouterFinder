import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# SAFE ROUTE FINDER - V1 MODEL SERVICE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL / SCALER / DATASET PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "safe_route_final_multibranch.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "safe_route_final_scalers.pkl"
)

TRAINING_DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "training",
    "route_safety_dataset_cleaned.csv"
)


# ============================================================
# CUSTOM KERAS LAYER
# IMPORTANT:
# The original V1 model was saved as SafeRoute>ScaleTo100
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="SafeRoute",
    name="ScaleTo100"
)
class ScaleTo100(tf.keras.layers.Layer):

    def call(self, inputs):
        return inputs * 100.0

    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# CHECK FILES
# ============================================================

print("\n==============================================")
print("SafeRoute V1 Model Service")
print("==============================================")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nV1 model not found:\n{MODEL_PATH}"
    )


if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"\nV1 scaler file not found:\n{SCALER_PATH}"
    )


if not os.path.exists(TRAINING_DATA_PATH):

    raise FileNotFoundError(
        f"\nTraining dataset not found:\n"
        f"{TRAINING_DATA_PATH}"
    )


# ============================================================
# LOAD V1 MODEL
# ============================================================

print("\nLoading SafeRoute V1 model...")

model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "ScaleTo100": ScaleTo100,
        "SafeRoute>ScaleTo100": ScaleTo100
    },

    compile=False
)

print("V1 model loaded successfully.")


# ============================================================
# LOAD SAVED SCALERS
# ============================================================

print("\nLoading V1 scalers...")

scalers = joblib.load(
    SCALER_PATH
)


traffic_scaler = scalers["traffic"]

lighting_scaler = scalers["lighting"]

weather_scaler = scalers["weather"]

crime_scaler = scalers["crime"]

emergency_scaler = scalers["emergency"]


print("V1 scalers loaded successfully.")


# ============================================================
# LOAD TRAINING DATA
#
# We need this because V1 calculated:
#
# rainfall_max
# visibility_min
# visibility_max
# crime_median
#
# directly from the training dataset.
# ============================================================

print("\nLoading V1 training dataset...")

training_df = pd.read_csv(
    TRAINING_DATA_PATH
)

print(
    "Training rows:",
    len(training_df)
)


# ============================================================
# V1 FEATURE-ENGINEERING CONSTANTS
# ============================================================

rainfall_max = max(

    float(
        training_df[
            "rainfall_mm"
        ].max()
    ),

    1.0
)


visibility_min = float(

    training_df[
        "visibility_km"
    ].min()
)


visibility_max = float(

    training_df[
        "visibility_km"
    ].max()
)


crime_median = max(

    float(
        training_df[
            "total_crime_count"
        ].median()
    ),

    1.0
)


print("\nV1 feature constants:")

print(
    "Rainfall max     :",
    rainfall_max
)

print(
    "Visibility min   :",
    visibility_min
)

print(
    "Visibility max   :",
    visibility_max
)

print(
    "Crime median     :",
    crime_median
)


# ============================================================
# V1 FEATURE ENGINEERING
# ============================================================

def prepare_v1_features(

    traffic_index,

    lighting_score,

    rainfall_mm,

    visibility_km,

    violent_crime_count,

    property_crime_count,

    women_crime_count,

    other_crime_count,

    total_crime_count,

    crime_rate_per_1000_exposure,

    nearest_police_km,

    nearest_hospital_km,

    emergency_access_score,

    month

):

    # ========================================================
    # TRAFFIC
    # ========================================================

    traffic_index = float(
        traffic_index
    )

    traffic_safety_score = np.clip(

        traffic_index,

        0,

        100
    )


    # ========================================================
    # LIGHTING
    # ========================================================

    lighting_score = np.clip(

        float(
            lighting_score
        ),

        0,

        100
    )


    # ========================================================
    # WEATHER
    # ========================================================

    rainfall_mm = max(

        float(
            rainfall_mm
        ),

        0.0
    )


    rainfall_risk = np.clip(

        (
            rainfall_mm
            /
            rainfall_max
            *
            100
        ),

        0,

        100
    )


    visibility_km = max(

        float(
            visibility_km
        ),

        0.0
    )


    if visibility_max > visibility_min:

        visibility_safety_score = (

            (
                visibility_km
                -
                visibility_min
            )
            /
            (
                visibility_max
                -
                visibility_min
            )
            *
            100

        )

    else:

        visibility_safety_score = 100.0


    visibility_safety_score = np.clip(

        visibility_safety_score,

        0,

        100
    )


    # ========================================================
    # MONTH
    # ========================================================

    month = int(month)


    month_sin = np.sin(

        2
        *
        np.pi
        *
        month
        /
        12

    )


    month_cos = np.cos(

        2
        *
        np.pi
        *
        month
        /
        12

    )


    # ========================================================
    # CRIME
    # ========================================================

    violent_crime_count = max(

        float(
            violent_crime_count
        ),

        0.0
    )


    property_crime_count = max(

        float(
            property_crime_count
        ),

        0.0
    )


    women_crime_count = max(

        float(
            women_crime_count
        ),

        0.0
    )


    other_crime_count = max(

        float(
            other_crime_count
        ),

        0.0
    )


    total_crime_count = max(

        float(
            total_crime_count
        ),

        0.0
    )


    crime_rate_per_1000_exposure = max(

        float(
            crime_rate_per_1000_exposure
        ),

        0.0
    )


    crime_intensity = (

        total_crime_count
        /
        crime_median

    )


    crime_pressure = (

        0.25
        *
        violent_crime_count

        +

        0.20
        *
        property_crime_count

        +

        0.20
        *
        women_crime_count

        +

        0.15
        *
        other_crime_count

        +

        0.20
        *
        crime_rate_per_1000_exposure

    )


    # ========================================================
    # EMERGENCY
    # ========================================================

    nearest_police_km = max(

        float(
            nearest_police_km
        ),

        0.0
    )


    nearest_hospital_km = max(

        float(
            nearest_hospital_km
        ),

        0.0
    )


    emergency_access_score = np.clip(

        float(
            emergency_access_score
        ),

        0,

        100
    )


    emergency_distance = (

        nearest_police_km
        +
        nearest_hospital_km

    )


    # ========================================================
    # FIVE V1 BRANCHES
    #
    # EXACTLY MATCHES TRAINING MODEL
    # ========================================================

    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    traffic = np.array(

        [
            traffic_index,
            traffic_safety_score
        ],

        dtype=np.float32

    ).reshape(

        1,
        -1
    )


    # --------------------------------------------------------
    # LIGHTING
    # --------------------------------------------------------

    lighting = np.array(

        [
            lighting_score
        ],

        dtype=np.float32

    ).reshape(

        1,
        -1
    )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather = np.array(

        [
            rainfall_mm,
            rainfall_risk,
            visibility_km,
            visibility_safety_score,
            month_sin,
            month_cos
        ],

        dtype=np.float32

    ).reshape(

        1,
        -1
    )


    # --------------------------------------------------------
    # CRIME
    # --------------------------------------------------------

    crime = np.array(

        [
            violent_crime_count,
            property_crime_count,
            women_crime_count,
            other_crime_count,
            total_crime_count,
            crime_rate_per_1000_exposure,
            crime_intensity,
            crime_pressure
        ],

        dtype=np.float32

    ).reshape(

        1,
        -1
    )


    # --------------------------------------------------------
    # EMERGENCY
    # --------------------------------------------------------

    emergency = np.array(

        [
            nearest_police_km,
            nearest_hospital_km,
            emergency_access_score,
            emergency_distance
        ],

        dtype=np.float32

    ).reshape(

        1,
        -1
    )


    return {

        "traffic": traffic,

        "lighting": lighting,

        "weather": weather,

        "crime": crime,

        "emergency": emergency

    }


# ============================================================
# SCALE V1 BRANCHES
# ============================================================

def scale_v1_features(
    features
):

    traffic_scaled = (

        traffic_scaler
        .transform(
            features["traffic"]
        )
        .astype(
            np.float32
        )

    )


    lighting_scaled = (

        lighting_scaler
        .transform(
            features["lighting"]
        )
        .astype(
            np.float32
        )

    )


    weather_scaled = (

        weather_scaler
        .transform(
            features["weather"]
        )
        .astype(
            np.float32
        )

    )


    crime_scaled = (

        crime_scaler
        .transform(
            features["crime"]
        )
        .astype(
            np.float32
        )

    )


    emergency_scaled = (

        emergency_scaler
        .transform(
            features["emergency"]
        )
        .astype(
            np.float32
        )

    )


    return {

        "traffic": traffic_scaled,

        "lighting": lighting_scaled,

        "weather": weather_scaled,

        "crime": crime_scaled,

        "emergency": emergency_scaled

    }


# ============================================================
# PREDICT SAFETY SCORE
# ============================================================

def predict_v1_safety(
    features
):

    scaled_features = scale_v1_features(
        features
    )


    prediction = model.predict(

        [

            scaled_features[
                "traffic"
            ],

            scaled_features[
                "lighting"
            ],

            scaled_features[
                "weather"
            ],

            scaled_features[
                "crime"
            ],

            scaled_features[
                "emergency"
            ]

        ],

        verbose=0
    )


    safety_score = float(

        np.asarray(
            prediction
        ).reshape(
            -1
        )[0]

    )


    safety_score = float(

        np.clip(

            safety_score,

            0,

            100

        )

    )


    return safety_score


# ============================================================
# SAFETY SCORE -> RISK LEVEL
# ============================================================

def get_risk_level(
    score
):

    score = float(
        score
    )


    if score < 30:

        return "Very High Risk"


    elif score < 45:

        return "High Risk"


    elif score < 60:

        return "Medium Risk"


    elif score < 75:

        return "Low Risk"


    else:

        return "Very Low Risk"


# ============================================================
# COMPLETE V1 PREDICTION
# ============================================================

def predict_route_safety(

    traffic_index,

    lighting_score,

    rainfall_mm,

    visibility_km,

    violent_crime_count,

    property_crime_count,

    women_crime_count,

    other_crime_count,

    total_crime_count,

    crime_rate_per_1000_exposure,

    nearest_police_km,

    nearest_hospital_km,

    emergency_access_score,

    month

):

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    features = prepare_v1_features(

        traffic_index=
            traffic_index,

        lighting_score=
            lighting_score,

        rainfall_mm=
            rainfall_mm,

        visibility_km=
            visibility_km,

        violent_crime_count=
            violent_crime_count,

        property_crime_count=
            property_crime_count,

        women_crime_count=
            women_crime_count,

        other_crime_count=
            other_crime_count,

        total_crime_count=
            total_crime_count,

        crime_rate_per_1000_exposure=
            crime_rate_per_1000_exposure,

        nearest_police_km=
            nearest_police_km,

        nearest_hospital_km=
            nearest_hospital_km,

        emergency_access_score=
            emergency_access_score,

        month=
            month

    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    safety_score = predict_v1_safety(
        features
    )


    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    risk_level = get_risk_level(
        safety_score
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "safety_score":
            round(
                safety_score,
                2
            ),

        "risk_level":
            risk_level

    }


# ============================================================
# TEST V1 MODEL
# ============================================================

if __name__ == "__main__":

    print(
        "\n=============================================="
    )

    print(
        "Testing SafeRoute V1..."
    )

    print(
        "=============================================="
    )


    test_result = predict_route_safety(

        # ----------------------------------------------------
        # Traffic
        # ----------------------------------------------------

        traffic_index=80,


        # ----------------------------------------------------
        # NASA lighting
        # ----------------------------------------------------

        lighting_score=70,


        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        rainfall_mm=0,

        visibility_km=10,


        # ----------------------------------------------------
        # Crime
        # ----------------------------------------------------

        violent_crime_count=200,

        property_crime_count=700,

        women_crime_count=250,

        other_crime_count=500,

        total_crime_count=1650,

        crime_rate_per_1000_exposure=8,


        # ----------------------------------------------------
        # Emergency
        # ----------------------------------------------------

        nearest_police_km=2,

        nearest_hospital_km=4,

        emergency_access_score=80,


        # ----------------------------------------------------
        # Current month
        # ----------------------------------------------------

        month=9

    )


    print(
        "\nV1 TEST RESULT"
    )

    print(
        "----------------------------------------------"
    )

    print(
        "Safety Score :",
        test_result[
            "safety_score"
        ]
    )

    print(
        "Risk Level   :",
        test_result[
            "risk_level"
        ]
    )

    print(
        "----------------------------------------------"
    )


    print(
        "\nV1 model test completed."
    )