import requests
import pandas as pd
import os
import time
from datetime import datetime


API_URL = "http://127.0.0.1:8000/routes"

OUTPUT_DIR = os.path.join("datasets", "training")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "route_training_features.csv")


# Andhra Pradesh route pairs
ROUTE_PAIRS = [
    ("Kadapa", "Tirupati"),
    ("Kadapa", "Nellore"),
    ("Kadapa", "Kurnool"),
    ("Kadapa", "Anantapur"),
    ("Kadapa", "Guntur"),
    ("Kadapa", "Vijayawada"),
    ("Kadapa", "Visakhapatnam"),

    ("Tirupati", "Nellore"),
    ("Tirupati", "Chittoor"),
    ("Tirupati", "Vijayawada"),
    ("Tirupati", "Guntur"),
    ("Tirupati", "Kadapa"),

    ("Nellore", "Guntur"),
    ("Nellore", "Vijayawada"),
    ("Nellore", "Tirupati"),
    ("Nellore", "Kadapa"),

    ("Vijayawada", "Guntur"),
    ("Vijayawada", "Visakhapatnam"),
    ("Vijayawada", "Rajahmundry"),
    ("Vijayawada", "Nellore"),
    ("Vijayawada", "Tirupati"),

    ("Guntur", "Vijayawada"),
    ("Guntur", "Nellore"),
    ("Guntur", "Ongole"),
    ("Guntur", "Visakhapatnam"),

    ("Visakhapatnam", "Vijayawada"),
    ("Visakhapatnam", "Rajahmundry"),
    ("Visakhapatnam", "Srikakulam"),
    ("Visakhapatnam", "Guntur"),

    ("Kurnool", "Anantapur"),
    ("Kurnool", "Kadapa"),
    ("Kurnool", "Vijayawada"),
    ("Kurnool", "Nandyal"),

    ("Anantapur", "Kadapa"),
    ("Anantapur", "Kurnool"),
    ("Anantapur", "Tirupati"),

    ("Rajahmundry", "Visakhapatnam"),
    ("Rajahmundry", "Vijayawada"),
    ("Rajahmundry", "Eluru"),

    ("Srikakulam", "Visakhapatnam"),
    ("Srikakulam", "Vijayawada"),

    ("Eluru", "Vijayawada"),
    ("Eluru", "Rajahmundry"),

    ("Ongole", "Guntur"),
    ("Ongole", "Nellore"),
]


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        number = float(value)

        if pd.isna(number):
            return default

        return number

    except Exception:
        return default


def extract_route_features(route):
    """
    Convert one API route response into a flat ML feature row.
    """

    lighting = route.get("lighting", {}) or {}
    traffic = route.get("traffic", {}) or {}
    weather = route.get("weather", {}) or {}
    crime = route.get("crime", {}) or {}

    row = {
        "distance_km": safe_float(route.get("distance_km")),
        "time_minutes": safe_float(route.get("time_minutes")),

        # NASA lighting
        "lighting_score": safe_float(
            lighting.get("lighting_score")
        ),

        "lighting_value": safe_float(
            lighting.get("lighting_value")
        ),

        "lighting_samples": safe_float(
            lighting.get("sample_count")
        ),

        # TomTom traffic
        "traffic_score": safe_float(
            traffic.get("traffic_score")
        ),

        "average_current_speed_kmph": safe_float(
            traffic.get("average_current_speed_kmph")
        ),

        "average_free_flow_speed_kmph": safe_float(
            traffic.get("average_free_flow_speed_kmph")
        ),

        "average_delay_seconds": safe_float(
            traffic.get("average_delay_seconds")
        ),

        "average_speed_ratio": safe_float(
            traffic.get("average_speed_ratio")
        ),

        "traffic_confidence": safe_float(
            traffic.get("traffic_confidence")
        ),

        "traffic_samples": safe_float(
            traffic.get("sample_count")
        ),

        # Open-Meteo weather
        "weather_score": safe_float(
            weather.get("weather_score")
        ),

        "average_temperature_c": safe_float(
            weather.get("average_temperature_c")
        ),

        "average_humidity_percent": safe_float(
            weather.get("average_humidity_percent")
        ),

        "average_precipitation_mm": safe_float(
            weather.get("average_precipitation_mm")
        ),

        "average_rain_mm": safe_float(
            weather.get("average_rain_mm")
        ),

        "average_wind_speed_kmph": safe_float(
            weather.get("average_wind_speed_kmph")
        ),

        "average_wind_gusts_kmph": safe_float(
            weather.get("average_wind_gusts_kmph")
        ),

        "average_cloud_cover_percent": safe_float(
            weather.get("average_cloud_cover_percent")
        ),

        "weather_samples": safe_float(
            weather.get("sample_count")
        ),

        # Crime
        "crime_risk_score": safe_float(
            crime.get("crime_risk_score")
        ),

        "crime_safety_score": safe_float(
            crime.get("crime_safety_score")
        ),

        "violent_crime_cases": safe_float(
            crime.get("violent_crime_cases")
        ),

        "property_crime_cases": safe_float(
            crime.get("property_crime_cases")
        ),

        "women_safety_crime_cases": safe_float(
            crime.get("women_safety_crime_cases")
        ),

        "total_crime_cases": safe_float(
            crime.get("total_crime_cases")
        ),

        "crime_samples": safe_float(
            crime.get("sample_count")
        ),
    }

    return row


def collect_routes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    print("=" * 70)
    print("SAFE ROUTE FINDER - TRAINING DATA GENERATOR")
    print("=" * 70)

    print(f"API: {API_URL}")
    print(f"Route pairs: {len(ROUTE_PAIRS)}")
    print()

    for index, (source, destination) in enumerate(ROUTE_PAIRS, start=1):

        print(
            f"[{index}/{len(ROUTE_PAIRS)}] "
            f"{source} -> {destination}"
        )

        payload = {
            "source": source,
            "destination": destination
        }

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=180
            )

            response.raise_for_status()

            data = response.json()

            routes = data.get("routes", [])

            if not routes:
                print("  WARNING: No routes returned.")
                continue

            route_batch_id = (
                f"{source}_{destination}_{index}"
            )

            collected_at = datetime.now().isoformat()

            for route in routes:

                row = extract_route_features(route)

                row["route_batch_id"] = route_batch_id
                row["source"] = source
                row["destination"] = destination
                row["route_id"] = route.get(
                    "route_id",
                    f"route_{len(all_rows) + 1}"
                )
                row["collected_at"] = collected_at

                all_rows.append(row)

            print(
                f"  SUCCESS: {len(routes)} route candidates collected."
            )

        except requests.exceptions.RequestException as e:

            print(
                f"  ERROR requesting API: {e}"
            )

        except Exception as e:

            print(
                f"  ERROR processing response: {e}"
            )

        time.sleep(1)

    if not all_rows:

        print()
        print("ERROR: No training data was collected.")
        print("Make sure the FastAPI server is running.")
        return

    df = pd.DataFrame(all_rows)

    # Put identifying columns first
    first_columns = [
        "route_batch_id",
        "source",
        "destination",
        "route_id",
        "collected_at",
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in first_columns
    ]

    df = df[first_columns + remaining_columns]

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)

    print(f"Rows collected: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()
    print(f"Saved to:")
    print(os.path.abspath(OUTPUT_FILE))

    print()
    print("Preview:")
    print(df.head())

    print()
    print("Missing values:")
    print(
        df.isna().sum()
        .sort_values(ascending=False)
        .head(10)
    )


if __name__ == "__main__":
    collect_routes()