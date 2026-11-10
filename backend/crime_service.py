import os
import math
import pandas as pd


# ============================================================
# FILE LOCATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CRIME_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "crime",
    "ap_district_city_crime_2026_project.csv"
)


# ============================================================
# LOAD CRIME DATASET
# ============================================================

if not os.path.exists(CRIME_DATASET):
    raise FileNotFoundError(
        f"Crime dataset not found:\n{CRIME_DATASET}"
    )

crime_df = pd.read_csv(CRIME_DATASET)

crime_df.columns = [
    column.strip()
    for column in crime_df.columns
]


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

for column in ["location_name", "district", "location_type", "data_type"]:
    if column in crime_df.columns:
        crime_df[column] = (
            crime_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "location_name",
    "location_type",
    "district",
    "crime_risk_score_0_100",
    "crime_safety_score_0_100"
]

missing_columns = [
    column
    for column in required_columns
    if column not in crime_df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in crime dataset: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "crime_risk_score_0_100",
    "crime_safety_score_0_100",
    "violent_crime_cases",
    "property_crime_cases",
    "women_safety_crime_cases",
    "total_ipc_crimes",
    "total_crime_cases"
]

for column in numeric_columns:
    if column in crime_df.columns:
        crime_df[column] = pd.to_numeric(
            crime_df[column],
            errors="coerce"
        ).fillna(0)


# ============================================================
# LOCATION COORDINATES
# ============================================================
#
# These are representative coordinates for matching route
# points to the nearest known city/district location.
#
# They are NOT crime-location coordinates.
#


LOCATION_COORDINATES = {

    # --------------------------------------------------------
    # MAJOR CITIES
    # --------------------------------------------------------

    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Tirupati": (13.6288, 79.4192),
    "Kurnool": (15.8281, 78.0373),
    "Nellore": (14.4426, 79.9865),
    "Kadapa": (14.4673, 78.8242),
    "Rajahmundry": (16.9891, 81.2293),
    "Anantapur": (14.6819, 77.6006),
    "Chittoor": (13.2172, 79.1003),
    "Eluru": (16.7107, 81.0952),
    "Ongole": (15.5057, 80.0499),
    "Machilipatnam": (16.1875, 81.1389),
    "Srikakulam": (18.2949, 83.8938),
    "Vizianagaram": (18.1067, 83.3956),
    "Nandyal": (15.4786, 78.4831),
    "Proddatur": (14.7502, 78.5482),
    "Amaravati": (16.5744, 80.3588),

    # --------------------------------------------------------
    # DISTRICT REPRESENTATIVE LOCATIONS
    # --------------------------------------------------------

    "Alluri Sitharama Raju": (17.6900, 82.9500),
    "Anakapalli": (17.6869, 83.0030),
    "Anantapur": (14.6819, 77.6006),
    "Bapatla": (15.9040, 80.4670),
    "Chittoor": (13.2172, 79.1003),
    "Dr. B. R. Ambedkar Konaseema": (16.6000, 82.0000),
    "East Godavari": (17.0000, 81.8000),
    "Eluru": (16.7107, 81.0952),
    "Guntur": (16.3067, 80.4365),
    "Kakinada": (16.9891, 82.2475),
    "Krishna": (16.5000, 80.7000),
    "Kurnool": (15.8281, 78.0373),
    "Nandyal": (15.4786, 78.4831),
    "Nellore": (14.4426, 79.9865),
    "NTR": (16.5200, 80.6500),
    "Palnadu": (16.1000, 79.7500),
    "Parvathipuram Manyam": (18.7800, 83.4300),
    "Prakasam": (15.5000, 79.9000),
    "Sri Potti Sriramulu Nellore": (14.4426, 79.9865),
    "Srikakulam": (18.2949, 83.8938),
    "Tirupati": (13.6288, 79.4192),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vizianagaram": (18.1067, 83.3956),
    "West Godavari": (16.7000, 81.3000),
    "YSR Kadapa": (14.4673, 78.8242),
}


# ============================================================
# NORMALIZE LOCATION NAME
# ============================================================

def normalize_name(name):
    if not name:
        return ""

    name = str(name).strip().lower()

    replacements = {
        "district": "",
        "city": "",
        "urban": "",
        "rural": "",
        "dr. b. r. ambedkar ": "",
        "dr b r ambedkar ": "",
        "ysr ": "",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = " ".join(name.split())

    return name


# ============================================================
# FIND COORDINATE
# ============================================================

def get_location_coordinates(location_name):

    normalized = normalize_name(location_name)

    # Exact / normalized match
    for name, coordinates in LOCATION_COORDINATES.items():

        if normalize_name(name) == normalized:
            return coordinates

    # Partial match
    for name, coordinates in LOCATION_COORDINATES.items():

        candidate = normalize_name(name)

        if (
            candidate in normalized
            or normalized in candidate
        ):
            return coordinates

    return None


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    latitude1,
    longitude1,
    latitude2,
    longitude2
):

    earth_radius_km = 6371.0

    lat1 = math.radians(latitude1)
    lat2 = math.radians(latitude2)

    delta_lat = math.radians(
        latitude2 - latitude1
    )

    delta_lon = math.radians(
        longitude2 - longitude1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# FIND NEAREST LOCATION
# ============================================================

def find_nearest_location(
    latitude,
    longitude,
    max_distance_km=50
):

    nearest_location = None
    nearest_distance = float("inf")

    for location_name, coordinates in LOCATION_COORDINATES.items():

        location_latitude = coordinates[0]
        location_longitude = coordinates[1]

        distance = haversine_distance(
            latitude,
            longitude,
            location_latitude,
            location_longitude
        )

        if distance < nearest_distance:

            nearest_distance = distance
            nearest_location = location_name

    if nearest_location is None:
        return None, None

    if nearest_distance > max_distance_km:
        return None, None

    return nearest_location, nearest_distance


# ============================================================
# FIND CRIME RECORD
# ============================================================

def get_crime_record(location_name):

    if not location_name:
        return None

    normalized = normalize_name(location_name)

    # --------------------------------------------------------
    # First preference: CITY
    # --------------------------------------------------------

    city_records = crime_df[
        crime_df["location_type"]
        .str.lower()
        .eq("city")
    ]

    for _, row in city_records.iterrows():

        if normalize_name(row["location_name"]) == normalized:

            return row.to_dict()

    # --------------------------------------------------------
    # Second preference: DISTRICT
    # --------------------------------------------------------

    district_records = crime_df[
        crime_df["location_type"]
        .str.lower()
        .eq("district")
    ]

    for _, row in district_records.iterrows():

        row_location = normalize_name(
            row["location_name"]
        )

        row_district = normalize_name(
            row["district"]
        )

        if (
            row_location == normalized
            or row_district == normalized
        ):
            return row.to_dict()

    return None


# ============================================================
# CALCULATE CRIME FOR ONE ROUTE POINT
# ============================================================

def get_point_crime(
    latitude,
    longitude
):

    location_name, distance = find_nearest_location(
        latitude,
        longitude
    )

    if location_name is None:

        return {
            "location_name": "Unknown",
            "location_type": "unknown",
            "distance_to_location_km": 0,
            "crime_risk_score": 0,
            "crime_safety_score": 0,
            "data_type": "unavailable"
        }

    record = get_crime_record(
        location_name
    )

    if record is None:

        return {
            "location_name": location_name,
            "location_type": "unknown",
            "distance_to_location_km": round(
                distance,
                2
            ),
            "crime_risk_score": 0,
            "crime_safety_score": 0,
            "data_type": "unavailable"
        }

    return {
        "location_name": record.get(
            "location_name",
            location_name
        ),

        "location_type": record.get(
            "location_type",
            "district"
        ),

        "district": record.get(
            "district",
            ""
        ),

        "distance_to_location_km": round(
            distance,
            2
        ),

        "crime_risk_score": round(
            float(
                record.get(
                    "crime_risk_score_0_100",
                    0
                )
            ),
            2
        ),

        "crime_safety_score": round(
            float(
                record.get(
                    "crime_safety_score_0_100",
                    0
                )
            ),
            2
        ),

        "violent_crime_cases": record.get(
            "violent_crime_cases",
            0
        ),

        "property_crime_cases": record.get(
            "property_crime_cases",
            0
        ),

        "women_safety_crime_cases": record.get(
            "women_safety_crime_cases",
            0
        ),

        "total_crime_cases": record.get(
            "total_crime_cases",
            record.get(
                "total_ipc_crimes",
                0
            )
        ),

        "data_type": record.get(
            "data_type",
            "unknown"
        )
    }


# ============================================================
# SAMPLE ROUTE POINTS
# ============================================================

def sample_route_points(
    coordinates,
    sample_count=10
):

    if not coordinates:
        return []

    if len(coordinates) <= sample_count:
        return coordinates

    indexes = []

    for i in range(sample_count):

        index = round(
            i * (len(coordinates) - 1)
            / (sample_count - 1)
        )

        indexes.append(index)

    return [
        coordinates[index]
        for index in indexes
    ]


# ============================================================
# CALCULATE CRIME FOR COMPLETE ROUTE
# ============================================================

def calculate_route_crime(
    coordinates
):

    if not coordinates:

        return {
            "crime_risk_score": 0,
            "crime_safety_score": 0,
            "crime_location": "Unknown",
            "crime_location_type": "unknown",
            "crime_data_type": "unavailable",
            "crime_sample_count": 0,
            "crime_locations": []
        }

    sample_points = sample_route_points(
        coordinates,
        sample_count=10
    )

    results = []

    for point in sample_points:

        longitude = point[0]
        latitude = point[1]

        try:

            crime = get_point_crime(
                latitude,
                longitude
            )

            if crime["data_type"] != "unavailable":
                results.append(crime)

        except Exception as error:

            print(
                "Crime lookup failed:",
                error
            )

    if not results:

        return {
            "crime_risk_score": 0,
            "crime_safety_score": 0,
            "crime_location": "Unknown",
            "crime_location_type": "unknown",
            "crime_data_type": "unavailable",
            "crime_sample_count": 0,
            "crime_locations": []
        }

    # ========================================================
    # AVERAGE CRIME RISK
    # ========================================================

    risk_scores = [
        item["crime_risk_score"]
        for item in results
    ]

    safety_scores = [
        item["crime_safety_score"]
        for item in results
    ]

    average_risk = sum(
        risk_scores
    ) / len(risk_scores)

    average_safety = sum(
        safety_scores
    ) / len(safety_scores)

    # ========================================================
    # COUNT CITY / DISTRICT SAMPLES
    # ========================================================

    city_samples = [
        item
        for item in results
        if item["location_type"].lower() == "city"
    ]

    district_samples = [
        item
        for item in results
        if item["location_type"].lower() == "district"
    ]

    # ========================================================
    # CITY HAS PRIORITY
    # ========================================================

    if city_samples:

        selected_locations = city_samples

        location_type = "city"

    else:

        selected_locations = district_samples

        location_type = "district"

    # Most frequently encountered location
    location_counts = {}

    for item in selected_locations:

        name = item["location_name"]

        location_counts[name] = (
            location_counts.get(name, 0) + 1
        )

    if location_counts:

        main_location = max(
            location_counts,
            key=location_counts.get
        )

    else:

        main_location = results[0][
            "location_name"
        ]

    # ========================================================
    # DATA TYPE
    # ========================================================

    data_types = [
        item["data_type"]
        for item in results
        if item.get("data_type")
    ]

    if any(
        "synthetic" in str(data_type).lower()
        or "estimate" in str(data_type).lower()
        for data_type in data_types
    ):

        route_data_type = "includes_estimated_data"

    else:

        route_data_type = data_types[0] \
            if data_types else "unknown"

    # ========================================================
    # UNIQUE LOCATIONS
    # ========================================================

    unique_locations = []

    for item in results:

        location = item["location_name"]

        if location not in unique_locations:

            unique_locations.append(
                location
            )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "crime_risk_score": round(
            max(
                0,
                min(
                    100,
                    average_risk
                )
            ),
            2
        ),

        "crime_safety_score": round(
            max(
                0,
                min(
                    100,
                    average_safety
                )
            ),
            2
        ),

        "crime_location": main_location,

        "crime_location_type": location_type,

        "crime_data_type": route_data_type,

        "crime_sample_count": len(results),

        "crime_locations": unique_locations
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SAFE ROUTE FINDER - CRIME SERVICE TEST")
    print("=" * 60)
    print()

    print(
        "Crime dataset:",
        CRIME_DATASET
    )

    print(
        "Records loaded:",
        len(crime_df)
    )

    print()

    # --------------------------------------------------------
    # Test Kadapa
    # --------------------------------------------------------

    latitude = 14.4673
    longitude = 78.8242

    print("Testing location:")
    print(
        f"Latitude: {latitude}"
    )
    print(
        f"Longitude: {longitude}"
    )
    print()

    result = get_point_crime(
        latitude,
        longitude
    )

    print("Crime Result")
    print("-" * 40)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 60)