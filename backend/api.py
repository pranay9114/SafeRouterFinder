from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests
import os
import math
from datetime import datetime
import time
import hashlib

import pandas as pd


# ============================================================
# EXISTING SAFEROUTE SERVICES
# ============================================================

from lighting_service import calculate_route_lighting
from traffic_service import calculate_route_traffic
from weather_service import calculate_route_weather
from crime_service import calculate_route_crime


# ============================================================
# V1 DEEP LEARNING MODEL
# ============================================================

from v1_model_service import predict_route_safety
from route_feedback_service import make_route_key, get_route_rating, save_route_rating
from location_share_service import router as location_share_router


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

app.include_router(location_share_router)


# ============================================================
# CORS
# ============================================================

FRONTEND_URL = os.getenv("FRONTEND_URL")

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *([FRONTEND_URL] if FRONTEND_URL else [])
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# URLs
# ============================================================

GRAPHOPPER_URL = os.getenv(
    "GRAPHOPPER_URL",
    "http://localhost:8989/route"
)

GRAPHHOPPER_API_KEY = os.getenv(
    "GRAPHHOPPER_API_KEY",
    ""
)

GEOCODING_URL = (
    "https://nominatim.openstreetmap.org/search"
)

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


TRAINING_DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "training",
    "route_safety_dataset_cleaned.csv"
)


CRIME_DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "crime",
    "ap_district_city_crime_2026_project.csv"
)


# ============================================================
# LOAD TRAINING DATA
#
# Used only for the emergency-access baseline because the
# trained V1 model expects:
#
# nearest_police_km
# nearest_hospital_km
# emergency_access_score
#
# These values are not currently returned by your live API.
# ============================================================

print("\nLoading SafeRoute training dataset...")

try:

    training_df = pd.read_csv(
        TRAINING_DATA_PATH
    )

    print(
        "Training dataset loaded:",
        len(training_df),
        "rows"
    )

except Exception as error:

    print(
        "Training dataset loading error:",
        error
    )

    training_df = pd.DataFrame()


# ============================================================
# LOAD PROJECT CRIME DATASET
#
# This contains the individual crime fields needed by V1.
# ============================================================

print("\nLoading SafeRoute crime dataset...")

try:

    crime_df = pd.read_csv(
        CRIME_DATA_PATH
    )

    print(
        "Crime dataset loaded:",
        len(crime_df),
        "rows"
    )

except Exception as error:

    print(
        "Crime dataset loading error:",
        error
    )

    crime_df = pd.DataFrame()


# ============================================================
# REQUEST MODEL
# ============================================================

class RouteRequest(BaseModel):

    source: str

    destination: str

    max_routes: int = 10

    excluded_route_keys: list[str] = []


class RouteRatingRequest(BaseModel):

    route_key: str
    rating: float
    comment: str = ""
    source: str = ""
    destination: str = ""


class RerouteRequest(BaseModel):

    latitude: float

    longitude: float

    destination_latitude: float

    destination_longitude: float


# ============================================================
# GEOCODING
# ============================================================

def get_coordinates(place_name):

    params = {

        "q":
            f"{place_name}, Andhra Pradesh, India",

        "format":
            "json",

        "limit":
            1
    }


    headers = {

        "User-Agent":
            "SafeRouteFinder/1.0"
    }


    try:

        response = requests.get(

            GEOCODING_URL,

            params=params,

            headers=headers,

            timeout=20

        )


        response.raise_for_status()


        data = response.json()


        if not data:

            return None


        latitude = float(
            data[0]["lat"]
        )


        longitude = float(
            data[0]["lon"]
        )


        print(

            f"Location found: "
            f"{place_name} -> "
            f"{latitude}, {longitude}"

        )


        return latitude, longitude


    except Exception as error:

        print(
            "Geocoding error:",
            error
        )

        return None


# ============================================================
# GRAPHHOPPER ROUTING
# ============================================================

def get_route(points):

    params = [

        (
            "profile",
            "car"
        ),

        (
            "points_encoded",
            "false"
        ),

        (
            "instructions",
            "true"
        ),

        (
            "locale",
            "en"
        ),

        (
            "via_point_instructions",
            "false"
        )

    ]


    for lat, lon in points:

        params.append(

            (
                "point",
                f"{lat},{lon}"
            )

        )


    # Hosted GraphHopper Directions API requires an API key.
    # Local GraphHopper does not require one.
    if "graphhopper.com" in GRAPHOPPER_URL and GRAPHHOPPER_API_KEY:
        params.append(

            (
                "key",
                GRAPHHOPPER_API_KEY
            )

        )


    try:

        response = requests.get(

            GRAPHOPPER_URL,

            params=params,

            timeout=120

        )


        if response.status_code != 200:

            print(
                "GraphHopper Error:",
                response.text
            )

            return None


        data = response.json()


        if not data.get("paths"):

            return None


        return data["paths"][0]


    except Exception as error:

        print(
            "GraphHopper connection error:",
            error
        )

        return None


# ============================================================
# ROUTE ENDPOINT SAFETY CHECK
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


def ensure_route_reaches_destination(route, destination):
    """
    GraphHopper should return a route ending at the final point.
    This guard makes that requirement explicit for our candidate routes.

    If a candidate route unexpectedly stops far away from the requested
    destination, calculate a connector from that final point to the real
    destination and append it to the route. This prevents the frontend
    from displaying an alternative route that appears to finish somewhere
    else.
    """

    if route is None:
        return None

    coordinates = (
        route.get("points", {})
        .get("coordinates", [])
    )

    if not coordinates:
        return route

    last_lon, last_lat = coordinates[-1]
    destination_lat, destination_lon = destination

    gap_km = haversine_km(
        last_lat,
        last_lon,
        destination_lat,
        destination_lon
    )

    print(
        f"Route endpoint gap to destination: {gap_km:.3f} km"
    )

    # GraphHopper normally snaps the requested destination to a nearby
    # road. Small differences are therefore normal. Only repair a route
    # when the endpoint is clearly far from the requested destination.
    MAX_ENDPOINT_GAP_KM = 1.5

    if gap_km <= MAX_ENDPOINT_GAP_KM:
        return route

    print(
        "WARNING: Route did not finish near the requested destination. "
        "Adding destination connector..."
    )

    connector = get_route([
        (last_lat, last_lon),
        destination
    ])

    if connector is None:
        print(
            "WARNING: Could not create destination connector. "
            "Keeping original route."
        )
        return route

    connector_coordinates = (
        connector.get("points", {})
        .get("coordinates", [])
    )

    if len(connector_coordinates) < 2:
        return route

    # Avoid duplicating the joining coordinate.
    combined_coordinates = (
        coordinates
        + connector_coordinates[1:]
    )

    route["points"]["coordinates"] = combined_coordinates

    route["distance"] = (
        float(route.get("distance", 0))
        + float(connector.get("distance", 0))
    )

    route["time"] = (
        int(route.get("time", 0))
        + int(connector.get("time", 0))
    )

    # The route now explicitly ends at the connector's destination.
    final_lon, final_lat = combined_coordinates[-1]
    final_gap_km = haversine_km(
        final_lat,
        final_lon,
        destination_lat,
        destination_lon
    )

    print(
        f"Destination connector added. Final endpoint gap: "
        f"{final_gap_km:.3f} km"
    )

    return route


# ============================================================
# CREATE ALTERNATIVE ROUTE CANDIDATES
# ============================================================

def create_candidate_points(route):
    """
    Generate strong lateral detour points so GraphHopper is much more
    likely to return genuinely different roads.

    The previous fast version used mostly latitude/longitude offsets that
    could move along the same corridor. This version calculates a rough
    perpendicular direction from the route itself and creates detours at
    20%, 35%, 50%, 65%, and 80% of the route.

    Maximum 12 candidates keeps the response much faster than the old
    40-candidate version while giving GraphHopper more opportunities to
    find 3 distinct routes.
    """

    coordinates = (
        route.get("points", {})
        .get("coordinates", [])
    )

    total_points = len(coordinates)

    if total_points < 20:
        return []

    # Use the overall route direction to calculate a perpendicular
    # direction. Coordinates are [longitude, latitude].
    start_lon, start_lat = coordinates[0]
    end_lon, end_lat = coordinates[-1]

    dx = end_lon - start_lon
    dy = end_lat - start_lat

    length = (dx * dx + dy * dy) ** 0.5

    if length == 0:
        return []

    # Perpendicular unit vector in degree-space.
    perp_lon = -dy / length
    perp_lat = dx / length

    candidates = []

    # Strong lateral offsets. Roughly 8-20+ km depending on latitude.
    # Both sides are tested so we can find roads on either side.
    candidate_patterns = [
        (0.12, 0.06), (0.12, -0.06),
        (0.20, 0.10), (0.20, -0.10),
        (0.28, 0.12), (0.28, -0.12),
        (0.35, 0.14), (0.35, -0.14),
        (0.43, 0.16), (0.43, -0.16),
        (0.50, 0.18), (0.50, -0.18),
        (0.57, 0.16), (0.57, -0.16),
        (0.65, 0.14), (0.65, -0.14),
        (0.72, 0.12), (0.72, -0.12),
        (0.80, 0.10), (0.80, -0.10),
        (0.88, 0.08), (0.88, -0.08),
        (0.50, 0.24), (0.50, -0.24),
        (0.35, 0.22), (0.35, -0.22),
        (0.65, 0.22), (0.65, -0.22),
    ]

    for position, offset in candidate_patterns:
        index = min(
            int(total_points * position),
            total_points - 1
        )

        lon, lat = coordinates[index]

        candidate_lon = lon + (perp_lon * offset)
        candidate_lat = lat + (perp_lat * offset)

        candidates.append(
            (candidate_lat, candidate_lon)
        )

    return candidates


# ============================================================
# ROUTE SIMILARITY
# ============================================================

def is_similar_route(
    route1,
    route2
):

    distance_difference = (

        abs(

            route1["distance"]
            -
            route2["distance"]

        )
        /
        1000

    )


    time_difference = (

        abs(

            route1["time"]
            -
            route2["time"]

        )
        /
        60000

    )


    if (

        distance_difference < 3

        and

        time_difference < 8

    ):

        return True


    return False


# ============================================================
# OPEN-METEO VISIBILITY
#
# V1 requires visibility_km.
#
# Open-Meteo provides visibility as a current weather variable.
# ============================================================

def get_current_visibility(
    latitude,
    longitude
):

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "current":
            "visibility",

        "timezone":
            "auto"

    }


    try:

        response = requests.get(

            OPEN_METEO_URL,

            params=params,

            timeout=20

        )


        response.raise_for_status()


        data = response.json()


        current = data.get(
            "current",
            {}
        )


        visibility_m = current.get(
            "visibility"
        )


        if visibility_m is None:

            return None


        visibility_km = (

            float(
                visibility_m
            )
            /
            1000.0

        )


        return max(
            visibility_km,
            0.0
        )


    except Exception as error:

        print(
            "Visibility error:",
            error
        )

        return None


# ============================================================
# ROUTE VISIBILITY
#
# Sample up to 5 route points.
# ============================================================

def calculate_route_visibility(
    coordinates
):

    if not coordinates:

        return {

            "visibility_km":
                10.0,

            "visibility_sample_count":
                0,

            "visibility_data_available":
                False

        }


    total_points = len(
        coordinates
    )


    sample_count = min(
        5,
        total_points
    )


    if sample_count == 1:

        indices = [0]

    else:

        indices = [

            int(

                i
                *
                (
                    total_points - 1
                )
                /
                (
                    sample_count - 1
                )

            )

            for i in range(
                sample_count
            )

        ]


    visibility_values = []


    for index in indices:

        lon, lat = coordinates[
            index
        ]


        visibility = get_current_visibility(

            lat,

            lon

        )


        if visibility is not None:

            visibility_values.append(
                visibility
            )


    if not visibility_values:

        return {

            "visibility_km":
                10.0,

            "visibility_sample_count":
                0,

            "visibility_data_available":
                False

        }


    return {

        "visibility_km":
            round(

                sum(
                    visibility_values
                )
                /
                len(
                    visibility_values
                ),

                2

            ),

        "visibility_sample_count":
            len(
                visibility_values
            ),

        "visibility_data_available":
            True

    }


# ============================================================
# NORMALIZE LOCATION NAME
# ============================================================

def normalize_name(value):

    if value is None:

        return ""

    return (

        str(value)
        .strip()
        .lower()

    )


# ============================================================
# FIND CRIME RECORD
#
# Uses the location identified by crime_service.py.
#
# Preference:
# 1. Exact city
# 2. Exact district
# 3. District field
# ============================================================

def get_crime_record(
    crime_result
):

    if crime_df.empty:

        return None


    location = normalize_name(

        crime_result.get(
            "crime_location"
        )

    )


    location_type = normalize_name(

        crime_result.get(
            "crime_location_type"
        )

    )


    locations = crime_result.get(
        "crime_locations",
        []
    )


    # --------------------------------------------------------
    # Collect possible names
    # --------------------------------------------------------

    possible_names = []


    if location:

        possible_names.append(
            location
        )


    if isinstance(
        locations,
        list
    ):

        for item in locations:

            if isinstance(
                item,
                str
            ):

                possible_names.append(
                    normalize_name(
                        item
                    )
                )

            elif isinstance(
                item,
                dict
            ):

                for key in [

                    "location_name",

                    "name",

                    "district",

                    "location"

                ]:

                    if key in item:

                        possible_names.append(

                            normalize_name(
                                item[key]
                            )

                        )


    # Remove duplicates

    possible_names = list(
        dict.fromkeys(
            possible_names
        )
    )


    # --------------------------------------------------------
    # Exact location match
    # --------------------------------------------------------

    for name in possible_names:

        if not name:

            continue


        matches = crime_df[

            crime_df[
                "location_name"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            name

        ]


        if not matches.empty:

            if location_type == "city":

                city_matches = matches[

                    matches[
                        "location_type"
                    ]
                    .astype(str)
                    .str.lower()
                    ==
                    "city"

                ]


                if not city_matches.empty:

                    return city_matches.iloc[0]


            return matches.iloc[0]


    # --------------------------------------------------------
    # District fallback
    # --------------------------------------------------------

    district = crime_result.get(
        "district"
    )


    if district:

        district_name = normalize_name(
            district
        )


        matches = crime_df[

            (
                crime_df[
                    "location_name"
                ]
                .astype(str)
                .str.lower()
                ==
                district_name
            )

            |

            (

                crime_df[
                    "district"
                ]
                .astype(str)
                .str.lower()
                ==
                district_name

            )

        ]


        if not matches.empty:

            district_rows = matches[

                matches[
                    "location_type"
                ]
                .astype(str)
                .str.lower()
                ==
                "district"

            ]


            if not district_rows.empty:

                return district_rows.iloc[0]


            return matches.iloc[0]


    return None


# ============================================================
# GET NUMERIC VALUE FROM CRIME RECORD
# ============================================================

def crime_value(
    record,
    column
):

    if record is None:

        return 0.0


    try:

        value = float(
            record[column]
        )


        if math.isnan(
            value
        ):

            return 0.0


        return max(
            value,
            0.0
        )


    except Exception:

        return 0.0


# ============================================================
# PREPARE V1 CRIME FEATURES
# ============================================================

def get_v1_crime_features(
    crime_result
):

    record = get_crime_record(
        crime_result
    )


    if record is None:

        print(
            "WARNING: No matching crime record "
            "found for V1."
        )


        return {

            "violent_crime_count":
                0.0,

            "property_crime_count":
                0.0,

            "women_crime_count":
                0.0,

            "other_crime_count":
                0.0,

            "total_crime_count":
                0.0,

            "crime_rate_per_1000_exposure":
                0.0,

            "crime_record_found":
                False

        }


    violent = crime_value(

        record,

        "violent_crime_cases"

    )


    property_crime = crime_value(

        record,

        "property_crime_cases"

    )


    women = crime_value(

        record,

        "women_safety_crime_cases"

    )


    total = crime_value(

        record,

        "total_ipc_crimes"

    )


    # --------------------------------------------------------
    # V1 needs "other crime".
    #
    # We calculate it from total minus the three main
    # categories, preventing negative values.
    # --------------------------------------------------------

    other = max(

        total
        -
        violent
        -
        property_crime
        -
        women,

        0.0

    )


    # --------------------------------------------------------
    # The project crime dataset does not contain the same
    # exposure-rate field used by the V1 training CSV.
    #
    # Therefore use a stable project-derived rate based on
    # total crime count rather than inventing a population.
    #
    # This is clearly labelled in the API response.
    # --------------------------------------------------------

    crime_rate = (

        total
        /
        1000.0

    )


    return {

        "violent_crime_count":
            violent,

        "property_crime_count":
            property_crime,

        "women_crime_count":
            women,

        "other_crime_count":
            other,

        "total_crime_count":
            total,

        "crime_rate_per_1000_exposure":
            crime_rate,

        "crime_record_found":
            True

    }


# ============================================================
# EMERGENCY FEATURES
#
# V1 requires:
# nearest_police_km
# nearest_hospital_km
# emergency_access_score
# emergency_distance
#
# Your current API has no live police/hospital service.
# We therefore use the district baseline from the V1 training
# dataset instead of inventing coordinates.
# ============================================================

def get_emergency_features(
    crime_result
):

    if training_df.empty:

        return {

            "nearest_police_km":
                5.0,

            "nearest_hospital_km":
                5.0,

            "emergency_access_score":
                50.0,

            "emergency_data_type":
                "fallback_baseline"

        }


    district = crime_result.get(
        "district"
    )


    if not district:

        district = crime_result.get(
            "crime_location"
        )


    district_name = normalize_name(
        district
    )


    if district_name:

        matches = training_df[

            training_df[
                "district"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            district_name

        ]

    else:

        matches = pd.DataFrame()


    # --------------------------------------------------------
    # If no exact district match exists, use overall training
    # median.
    # --------------------------------------------------------

    if matches.empty:

        nearest_police = float(

            training_df[
                "nearest_police_km"
            ].median()

        )


        nearest_hospital = float(

            training_df[
                "nearest_hospital_km"
            ].median()

        )


        emergency_score = float(

            training_df[
                "emergency_access_score"
            ].median()

        )


        data_type = (
            "training_dataset_overall_baseline"
        )


    else:

        nearest_police = float(

            matches[
                "nearest_police_km"
            ].median()

        )


        nearest_hospital = float(

            matches[
                "nearest_hospital_km"
            ].median()

        )


        emergency_score = float(

            matches[
                "emergency_access_score"
            ].median()

        )


        data_type = (
            "training_dataset_district_baseline"
        )


    return {

        "nearest_police_km":
            round(
                max(
                    nearest_police,
                    0
                ),
                2
            ),

        "nearest_hospital_km":
            round(
                max(
                    nearest_hospital,
                    0
                ),
                2
            ),

        "emergency_access_score":
            round(

                min(

                    max(
                        emergency_score,
                        0
                    ),

                    100

                ),

                2

            ),

        "emergency_data_type":
            data_type

    }


# ============================================================
# EXTRACT ROUTE CORRIDOR / HIGHWAY NAME
# ============================================================

def extract_route_corridor_name(route, source_name="", dest_name=""):
    """
    Extracts a human-readable highway/corridor name summary from GraphHopper instructions.
    Example: 'via NH16', 'via NH16 / AH45', 'via Vijayawada Bypass'
    """
    instructions = route.get("instructions", [])
    road_dist = {}

    for inst in instructions:
        if int(inst.get("sign", 0) or 0) == 5:
            continue
        street = (inst.get("street_name") or "").strip()
        if not street or street.lower() in ["unnamed", "unnamed road", "-", "service road", "ramp"]:
            continue
        dist = float(inst.get("distance", 0) or 0)
        road_dist[street] = road_dist.get(street, 0.0) + dist

    if not road_dist:
        import re
        for inst in instructions:
            text = inst.get("text", "")
            m = re.search(r'\b(?:onto|on|along)\s+([A-Z0-9\s\-/]+)', text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if cand and cand.lower() not in ["the road", "unnamed road"]:
                    road_dist[cand] = road_dist.get(cand, 0.0) + float(inst.get("distance", 0) or 0)

    if not road_dist:
        if route.get("description"):
            desc = str(route.get("description")).strip()
            return desc if desc.lower().startswith("via") else f"via {desc}"
        if source_name and dest_name:
            return f"via {source_name} - {dest_name} Corridor"
        return "via Main Corridor"

    sorted_roads = sorted(road_dist.items(), key=lambda x: x[1], reverse=True)
    top_road = sorted_roads[0][0]
    total_dist = float(route.get("distance", 0) or 0) or sum(road_dist.values()) or 1.0

    selected = [top_road]
    if len(sorted_roads) > 1:
        second_road = sorted_roads[1][0]
        second_dist = sorted_roads[1][1]
        if second_dist >= 5000 and (second_dist / total_dist) >= 0.15:
            if second_road.lower() not in top_road.lower() and top_road.lower() not in second_road.lower():
                selected.append(second_road)

    summary = " / ".join(selected)
    if not summary.lower().startswith("via"):
        summary = f"via {summary}"
    return summary


# ============================================================
# SAFETY RANK
# ============================================================

def get_ranked_routes(
    routes
):
    """Rank primarily by AI safety, with historical user ratings as a secondary signal."""
    for route in routes:
        safety = float(route.get("safety_score", 0) or 0)
        rating_count = int(route.get("rating_count", 0) or 0)
        rating_score = float(route.get("rating_score", 0) or 0)

        # Ratings should help, but must not overpower the safety model.
        # Influence grows with feedback and caps at 20%.
        rating_weight = 0.0 if rating_count == 0 else min(0.20, 0.05 + (rating_count * 0.03))
        ranking_score = safety * (1.0 - rating_weight) + rating_score * rating_weight
        route["rating_weight"] = round(rating_weight, 3)
        route["ranking_score"] = round(ranking_score, 2)

    sorted_routes = sorted(
        routes,
        key=lambda route: (route.get("ranking_score", 0), route.get("safety_score", 0)),
        reverse=True
    )

    for rank, route in enumerate(sorted_routes, start=1):
        route["safety_rank"] = rank
        route["recommended"] = rank == 1

    return sorted_routes


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Safe Route Finder API is running!",

        "model":
            "SafeRoute V1 Multi-Branch Deep Learning",

        "status":
            "ready"

    }


# ============================================================
# MAIN ROUTE API
# ============================================================

@app.post("/routes")
def generate_routes(
    request: RouteRequest
):

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "SAFE ROUTE REQUEST RECEIVED"
    )


    print(
        "Source:",
        request.source
    )


    print(
        "Destination:",
        request.destination
    )


    print(
        "=" * 70
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not request.source.strip():

        return {

            "success":
                False,

            "message":
                "Please enter a source location."

        }


    if not request.destination.strip():

        return {

            "success":
                False,

            "message":
                "Please enter a destination location."

        }


    # ========================================================
    # SOURCE
    # ========================================================

    source = get_coordinates(

        request.source.strip()

    )


    if source is None:

        return {

            "success":
                False,

            "message":
                (
                    "Could not find source location: "
                    +
                    request.source
                )

        }


    # ========================================================
    # DESTINATION
    # ========================================================

    destination = get_coordinates(

        request.destination.strip()

    )


    if destination is None:

        return {

            "success":
                False,

            "message":
                (
                    "Could not find destination location: "
                    +
                    request.destination
                )

        }


    print(
        "\nSource coordinates:",
        source
    )


    print(
        "Destination coordinates:",
        destination
    )

    target_route_count = min(max(int(request.max_routes or 10), 1), 10)
    excluded_route_keys = set(request.excluded_route_keys or [])

    print(f"Requested route limit: {target_route_count}")
    print(f"Excluded existing routes: {len(excluded_route_keys)}")


    # ========================================================
    # MAIN ROUTE
    # ========================================================

    print(
        "\nFinding main route..."
    )


    main_route = get_route([

        source,

        destination

    ])


    if main_route is None:

        return {

            "success":
                False,

            "message":
                (
                    "Could not find a road route "
                    "between these locations."
                )

        }


    print(
        "Main route found!"
    )


    # Validate the main route endpoint as well.
    main_route = ensure_route_reaches_destination(

        main_route,

        destination

    )


    main_route_key = make_route_key(request.source, request.destination, main_route.get("points", {}).get("coordinates", []))

    routes = [] if main_route_key in excluded_route_keys else [main_route]

    if main_route_key in excluded_route_keys:
        print("Main route already known by the client; searching for additional routes.")


    # ========================================================
    # ALTERNATIVE ROUTES
    # ========================================================

    print(
        "\nGenerating alternative routes..."
    )


    candidates = create_candidate_points(

        main_route

    )


    print(

        f"Generated "
        f"{len(candidates)} "
        f"candidate detour points."

    )


    # Allow enough candidate attempts to discover up to 10 distinct routes.
    candidates = candidates[:26]

    for candidate_number, candidate in enumerate(candidates, start=1):

        print(
            f"Trying route candidate "
            f"{candidate_number}/{len(candidates)}..."
        )

        if len(routes) >= target_route_count:
            break


        candidate_route = get_route([

            source,

            candidate,

            destination

        ])


        if candidate_route is None:

            continue


        # IMPORTANT: every candidate must finish at the SAME requested
        # destination. If GraphHopper ever returns an endpoint that is
        # clearly elsewhere, repair it before accepting the route.
        candidate_route = ensure_route_reaches_destination(

            candidate_route,

            destination

        )


        if candidate_route is None:

            continue

        candidate_coordinates = candidate_route.get("points", {}).get("coordinates", [])
        candidate_route_key = make_route_key(request.source, request.destination, candidate_coordinates)

        if candidate_route_key in excluded_route_keys:
            print("Skipping route already shown to the user.")
            continue

        duplicate = False


        for existing_route in routes:

            if is_similar_route(

                candidate_route,

                existing_route

            ):

                duplicate = True

                break


        if not duplicate:

            routes.append(

                candidate_route

            )


            distance_km = (

                candidate_route[
                    "distance"
                ]
                /
                1000

            )


            time_minutes = (

                candidate_route[
                    "time"
                ]
                /
                60000

            )


            print(

                f"Alternative route found: "
                f"{distance_km:.2f} km, "
                f"{time_minutes:.2f} minutes"

            )


    # ========================================================
    # ANALYZE EACH ROUTE
    # ========================================================

    result = []


    for i, route in enumerate(

        routes,

        start=1

    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"ROUTE {i}"
        )

        print(
            "=" * 70
        )


        coordinates = route[
            "points"
        ][
            "coordinates"
        ]


        # ====================================================
        # NASA NIGHTTIME LIGHTING
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Calculating NASA nighttime lighting..."
        )


        try:

            lighting = calculate_route_lighting(

                coordinates

            )


        except Exception as error:

            print(

                f"Lighting error Route {i}:",

                error

            )


            lighting = {

                "lighting_value":
                    0,

                "lighting_score":
                    0,

                "sample_count":
                    0

            }


        lighting_score = float(

            lighting.get(
                "lighting_score",
                0
            )

        )


        print(

            f"Lighting Score: "
            f"{lighting_score:.2f}/100"

        )


        # ====================================================
        # TOMTOM TRAFFIC
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Calculating TomTom traffic..."
        )


        try:

            traffic = calculate_route_traffic(

                coordinates

            )


        except Exception as error:

            print(

                f"Traffic error Route {i}:",

                error

            )


            traffic = {

                "traffic_score":
                    0,

                "average_current_speed_kmph":
                    0,

                "average_free_flow_speed_kmph":
                    0,

                "average_delay_seconds":
                    0,

                "average_speed_ratio":
                    0,

                "average_confidence":
                    0,

                "road_closure_count":
                    0,

                "traffic_sample_count":
                    0

            }


        traffic_score = float(

            traffic.get(
                "traffic_score",
                0
            )

        )


        print(

            f"Traffic Score: "
            f"{traffic_score:.2f}/100"

        )


        # ====================================================
        # OPEN-METEO WEATHER
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Calculating Open-Meteo weather..."
        )


        try:

            weather = calculate_route_weather(

                coordinates

            )


        except Exception as error:

            print(

                f"Weather error Route {i}:",

                error

            )


            weather = {

                "weather_score":
                    0,

                "weather_condition":
                    "Unavailable",

                "average_temperature_c":
                    0,

                "average_humidity_percent":
                    0,

                "average_precipitation_mm":
                    0,

                "average_rain_mm":
                    0,

                "average_wind_speed_kmph":
                    0,

                "average_wind_gusts_kmph":
                    0,

                "average_cloud_cover_percent":
                    0,

                "weather_sample_count":
                    0

            }


        rainfall_mm = float(

            weather.get(

                "average_precipitation_mm",

                0

            )

        )


        weather_score = float(

            weather.get(

                "weather_score",

                0

            )

        )


        print(

            f"Weather Score: "
            f"{weather_score:.2f}/100"

        )


        # ====================================================
        # VISIBILITY
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Calculating Open-Meteo visibility..."
        )


        visibility = calculate_route_visibility(

            coordinates

        )


        visibility_km = float(

            visibility[
                "visibility_km"
            ]

        )


        print(

            f"Visibility: "
            f"{visibility_km:.2f} km"

        )


        # ====================================================
        # CRIME
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Calculating crime..."
        )


        try:

            crime = calculate_route_crime(

                coordinates

            )


        except Exception as error:

            print(

                f"Crime error Route {i}:",

                error

            )


            crime = {

                "crime_risk_score":
                    0,

                "crime_safety_score":
                    0,

                "crime_location":
                    "Unknown",

                "crime_location_type":
                    "unknown",

                "crime_data_type":
                    "unavailable",

                "crime_sample_count":
                    0,

                "crime_locations":
                    []

            }


        print(

            f"Crime Risk: "
            f"{crime.get('crime_risk_score', 0)}/100"

        )


        print(

            f"Crime Safety: "
            f"{crime.get('crime_safety_score', 0)}/100"

        )


        print(

            "Crime Location:",
            crime.get(
                "crime_location",
                "Unknown"
            )

        )


        # ====================================================
        # V1 CRIME FEATURES
        # ====================================================

        v1_crime = get_v1_crime_features(

            crime

        )


        print(
            "\nV1 crime features:"
        )


        print(

            "Violent:",
            v1_crime[
                "violent_crime_count"
            ]

        )


        print(

            "Property:",
            v1_crime[
                "property_crime_count"
            ]

        )


        print(

            "Women:",
            v1_crime[
                "women_crime_count"
            ]

        )


        print(

            "Other:",
            v1_crime[
                "other_crime_count"
            ]

        )


        print(

            "Total:",
            v1_crime[
                "total_crime_count"
            ]

        )


        # ====================================================
        # EMERGENCY
        # ====================================================

        emergency = get_emergency_features(

            crime

        )


        print(
            "\nEmergency features:"
        )


        print(

            "Nearest Police:",
            emergency[
                "nearest_police_km"
            ],
            "km"

        )


        print(

            "Nearest Hospital:",
            emergency[
                "nearest_hospital_km"
            ],
            "km"

        )


        print(

            "Emergency Access:",
            emergency[
                "emergency_access_score"
            ],
            "/100"

        )


        # ====================================================
        # V1 DEEP LEARNING PREDICTION
        # ====================================================

        print(
            f"\n[Route {i}] "
            "Running SafeRoute V1 deep-learning model..."
        )


        try:

            current_month = datetime.now().month


            v1_prediction = predict_route_safety(

                traffic_index=
                    traffic_score,

                lighting_score=
                    lighting_score,

                rainfall_mm=
                    rainfall_mm,

                visibility_km=
                    visibility_km,

                violent_crime_count=
                    v1_crime[
                        "violent_crime_count"
                    ],

                property_crime_count=
                    v1_crime[
                        "property_crime_count"
                    ],

                women_crime_count=
                    v1_crime[
                        "women_crime_count"
                    ],

                other_crime_count=
                    v1_crime[
                        "other_crime_count"
                    ],

                total_crime_count=
                    v1_crime[
                        "total_crime_count"
                    ],

                crime_rate_per_1000_exposure=
                    v1_crime[
                        "crime_rate_per_1000_exposure"
                    ],

                nearest_police_km=
                    emergency[
                        "nearest_police_km"
                    ],

                nearest_hospital_km=
                    emergency[
                        "nearest_hospital_km"
                    ],

                emergency_access_score=
                    emergency[
                        "emergency_access_score"
                    ],

                month=
                    current_month

            )


            safety_score = float(

                v1_prediction[
                    "safety_score"
                ]

            )


            risk_level = (

                v1_prediction[
                    "risk_level"
                ]

            )


        except Exception as error:

            print(

                f"V1 model error Route {i}:",

                error

            )


            safety_score = 0.0

            risk_level = "Unavailable"


        print()
        print(
            "------------------------------------------"
        )


        print(
            f"V1 SAFETY SCORE: "
            f"{safety_score:.2f}/100"
        )


        print(
            f"V1 RISK LEVEL: "
            f"{risk_level}"
        )


        print(
            "------------------------------------------"
        )


        # ====================================================
        # ROUTE RESPONSE
        # ====================================================

        route_data = {

            # ------------------------------------------------
            # ROUTE
            # ------------------------------------------------

            "route_id":
                i,

            "name":
                extract_route_corridor_name(
                    route,
                    request.source,
                    request.destination
                ),

            "route_name":
                extract_route_corridor_name(
                    route,
                    request.source,
                    request.destination
                ),

            "corridor_name":
                extract_route_corridor_name(
                    route,
                    request.source,
                    request.destination
                ),

            "distance_km":
                round(

                    route[
                        "distance"
                    ]
                    /
                    1000,

                    2

                ),

            "time_minutes":
                round(

                    route[
                        "time"
                    ]
                    /
                    60000,

                    2

                ),


            # ------------------------------------------------
            # NASA
            # ------------------------------------------------

            "lighting_value":
                lighting.get(
                    "lighting_value",
                    0
                ),

            "lighting_score":
                round(
                    lighting_score,
                    2
                ),

            "lighting_sample_count":
                lighting.get(
                    "sample_count",
                    0
                ),


            # ------------------------------------------------
            # TRAFFIC
            # ------------------------------------------------

            "traffic_score":
                round(
                    traffic_score,
                    2
                ),

            "average_current_speed_kmph":
                traffic.get(
                    "average_current_speed_kmph",
                    0
                ),

            "average_free_flow_speed_kmph":
                traffic.get(
                    "average_free_flow_speed_kmph",
                    0
                ),

            "average_delay_seconds":
                traffic.get(
                    "average_delay_seconds",
                    0
                ),

            "average_speed_ratio":
                traffic.get(
                    "average_speed_ratio",
                    0
                ),

            "traffic_confidence":
                traffic.get(
                    "average_confidence",
                    0
                ),

            "road_closure_count":
                traffic.get(
                    "road_closure_count",
                    0
                ),

            "traffic_sample_count":
                traffic.get(
                    "traffic_sample_count",
                    0
                ),


            # ------------------------------------------------
            # WEATHER
            # ------------------------------------------------

            "weather_score":
                round(
                    weather_score,
                    2
                ),

            "weather_condition":
                weather.get(
                    "weather_condition",
                    "Unknown"
                ),

            "average_temperature_c":
                weather.get(
                    "average_temperature_c",
                    0
                ),

            "average_humidity_percent":
                weather.get(
                    "average_humidity_percent",
                    0
                ),

            "average_precipitation_mm":
                weather.get(
                    "average_precipitation_mm",
                    0
                ),

            "average_rain_mm":
                weather.get(
                    "average_rain_mm",
                    0
                ),

            "average_wind_speed_kmph":
                weather.get(
                    "average_wind_speed_kmph",
                    0
                ),

            "average_wind_gusts_kmph":
                weather.get(
                    "average_wind_gusts_kmph",
                    0
                ),

            "average_cloud_cover_percent":
                weather.get(
                    "average_cloud_cover_percent",
                    0
                ),

            "weather_sample_count":
                weather.get(
                    "weather_sample_count",
                    0
                ),


            # ------------------------------------------------
            # VISIBILITY
            # ------------------------------------------------

            "visibility_km":
                visibility_km,

            "visibility_sample_count":
                visibility[
                    "visibility_sample_count"
                ],

            "visibility_data_available":
                visibility[
                    "visibility_data_available"
                ],


            # ------------------------------------------------
            # CRIME
            # ------------------------------------------------

            "crime_risk_score":
                crime.get(
                    "crime_risk_score",
                    0
                ),

            "crime_safety_score":
                crime.get(
                    "crime_safety_score",
                    0
                ),

            "crime_location":
                crime.get(
                    "crime_location",
                    "Unknown"
                ),

            "crime_location_type":
                crime.get(
                    "crime_location_type",
                    "unknown"
                ),

            "crime_data_type":
                crime.get(
                    "crime_data_type",
                    "unknown"
                ),

            "crime_sample_count":
                crime.get(
                    "crime_sample_count",
                    0
                ),

            "crime_locations":
                crime.get(
                    "crime_locations",
                    []
                ),


            # ------------------------------------------------
            # V1 CRIME FEATURES
            # ------------------------------------------------

            "violent_crime_count":
                v1_crime[
                    "violent_crime_count"
                ],

            "property_crime_count":
                v1_crime[
                    "property_crime_count"
                ],

            "women_crime_count":
                v1_crime[
                    "women_crime_count"
                ],

            "other_crime_count":
                v1_crime[
                    "other_crime_count"
                ],

            "total_crime_count":
                v1_crime[
                    "total_crime_count"
                ],

            "crime_rate_per_1000_exposure":
                v1_crime[
                    "crime_rate_per_1000_exposure"
                ],


            # ------------------------------------------------
            # EMERGENCY
            # ------------------------------------------------

            "nearest_police_km":
                emergency[
                    "nearest_police_km"
                ],

            "nearest_hospital_km":
                emergency[
                    "nearest_hospital_km"
                ],

            "emergency_access_score":
                emergency[
                    "emergency_access_score"
                ],

            "emergency_data_type":
                emergency[
                    "emergency_data_type"
                ],


            # ------------------------------------------------
            # V1 FINAL DEEP LEARNING RESULT
            # ------------------------------------------------

            "safety_score":
                round(
                    safety_score,
                    2
                ),

            "risk_level":
                risk_level,

            "model":
                "SafeRoute V1 Multi-Branch Deep Learning",


            # ------------------------------------------------
            # USER RATINGS / LEARNING SIGNAL
            # ------------------------------------------------

            "route_key":
                make_route_key(
                    request.source,
                    request.destination,
                    coordinates
                ),

            **get_route_rating(
                make_route_key(
                    request.source,
                    request.destination,
                    coordinates
                )
            ),

            # ------------------------------------------------
            # ROUTE COORDINATES
            # ------------------------------------------------

            "coordinates":
                coordinates,

            # ------------------------------------------------
            # TURN-BY-TURN NAVIGATION
            # ------------------------------------------------

            "navigation_instructions": [

                {

                    "text":
                        instruction.get(
                            "text",
                            ""
                        ),

                    "street_name":
                        instruction.get(
                            "street_name",
                            ""
                        ),

                    "distance_m":
                        round(
                            float(
                                instruction.get(
                                    "distance",
                                    0
                                )
                            ),
                            1
                        ),

                    "time_ms":
                        int(
                            instruction.get(
                                "time",
                                0
                            )
                        ),

                    "sign":
                        int(
                            instruction.get(
                                "sign",
                                0
                            )
                        ),

                    "interval":
                        instruction.get(
                            "interval",
                            [0, 0]
                        ),

                    "exit_number":
                        instruction.get(
                            "exit_number"
                        )

                }

                for instruction in route.get(
                    "instructions",
                    []
                )

                # Hide GraphHopper's artificial via-point
                # instructions from the user-facing navigator.
                if int(
                    instruction.get(
                        "sign",
                        0
                    )
                ) != 5

            ]

        }


        result.append(
            route_data
        )


    # ========================================================
    # RANK ROUTES USING V1 SAFETY SCORE
    # ========================================================

    result = get_ranked_routes(
        result
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "FINAL SAFEROUTE RANKING"
    )


    print(
        "=" * 70
    )


    for route in result:

        print(

            f"Rank {route['safety_rank']} "
            f"| Route {route['route_id']} "
            f"| Safety "
            f"{route['safety_score']:.2f} "
            f"| {route['risk_level']}"

        )


    print(
        "=" * 70
    )


    return {

        "success":
            True,


        "model":
            "SafeRoute V1 Multi-Branch Deep Learning",


        "source_coordinates": {

            "latitude":
                source[0],

            "longitude":
                source[1]

        },


        "destination_coordinates": {

            "latitude":
                destination[0],

            "longitude":
                destination[1]

        },


        "recommended_route_id":

            result[0][
                "route_id"
            ]
            if result
            else None,


        "routes":
            result

    }


# ============================================================
# LIVE SAFETY CHECKPOINT API
#
# Called while the user is moving.  Unlike /routes, this endpoint
# evaluates the user's CURRENT location instead of averaging the
# entire selected route.
# ============================================================

class LiveSafetyRequest(BaseModel):
    latitude: float
    longitude: float


def _live_feature_fallbacks():
    return {
        "lighting": {
            "lighting_value": 0,
            "lighting_score": 0,
            "sample_count": 0
        },
        "traffic": {
            "traffic_score": 0,
            "average_current_speed_kmph": 0,
            "average_free_flow_speed_kmph": 0,
            "average_delay_seconds": 0,
            "average_speed_ratio": 0,
            "average_confidence": 0,
            "road_closure_count": 0,
            "traffic_sample_count": 0
        },
        "weather": {
            "weather_score": 0,
            "weather_condition": "Unavailable",
            "average_temperature_c": 0,
            "average_humidity_percent": 0,
            "average_precipitation_mm": 0,
            "average_rain_mm": 0,
            "average_wind_speed_kmph": 0,
            "average_wind_gusts_kmph": 0,
            "average_cloud_cover_percent": 0,
            "weather_sample_count": 0
        },
        "crime": {
            "crime_risk_score": 0,
            "crime_safety_score": 0,
            "crime_location": "Unknown",
            "crime_location_type": "unknown",
            "crime_data_type": "unavailable",
            "crime_sample_count": 0,
            "crime_locations": []
        }
    }


@app.post("/rate-route")
def rate_route(request: RouteRatingRequest):
    try:
        result = save_route_rating(
            request.route_key,
            request.rating,
            request.comment,
            request.source,
            request.destination
        )
        return {
            "success": True,
            "message": "Route rating saved.",
            "route_key": request.route_key,
            **result
        }
    except Exception as error:
        return {
            "success": False,
            "message": str(error)
        }


@app.post("/reroute")
def reroute(request: RerouteRequest):
    """
    Google-Maps-style rerouting:
    calculate a fresh road route from the user's current GPS
    position to the original destination.
    """

    values = [
        float(request.latitude),
        float(request.longitude),
        float(request.destination_latitude),
        float(request.destination_longitude)
    ]

    if not all(math.isfinite(value) for value in values):
        return {
            "success": False,
            "message": "Invalid coordinates."
        }

    current = (values[0], values[1])
    destination = (values[2], values[3])

    print(
        "\nREROUTE REQUEST"
        f"\nCurrent: {current}"
        f"\nDestination: {destination}"
    )

    route = get_route([
        current,
        destination
    ])

    if route is None:
        return {
            "success": False,
            "message": "GraphHopper could not calculate a new route."
        }

    route = ensure_route_reaches_destination(
        route,
        destination
    )

    instructions = [
        {
            "text": instruction.get("text", ""),
            "street_name": instruction.get("street_name", ""),
            "distance_m": round(
                float(instruction.get("distance", 0) or 0),
                1
            ),
            "time_ms": int(
                instruction.get("time", 0) or 0
            ),
            "sign": int(
                instruction.get("sign", 0) or 0
            ),
            "interval": instruction.get(
                "interval",
                [0, 0]
            ),
            "exit_number": instruction.get(
                "exit_number"
            )
        }
        for instruction in route.get(
            "instructions",
            []
        )
        if int(
            instruction.get("sign", 0) or 0
        ) != 5
    ]

    return {
        "success": True,
        "route": {
            "name": extract_route_corridor_name(route),
            "corridor_name": extract_route_corridor_name(route),
            "distance_km": round(
                float(route.get("distance", 0) or 0) / 1000,
                2
            ),
            "time_minutes": round(
                float(route.get("time", 0) or 0) / 60000,
                2
            ),
            "coordinates": route.get(
                "points",
                {}
            ).get(
                "coordinates",
                []
            ),
            "navigation_instructions": instructions
        }
    }


@app.post("/live-safety")
def live_safety(request: LiveSafetyRequest):
    """Return current-area safety information for live navigation."""

    latitude = float(request.latitude)
    longitude = float(request.longitude)

    # Keep coordinates inside valid geographic bounds.
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {
            "success": False,
            "message": "Invalid GPS coordinates."
        }

    point = [[longitude, latitude]]
    fallbacks = _live_feature_fallbacks()

    print(
        f"\\nLIVE SAFETY CHECK -> {latitude:.6f}, {longitude:.6f}"
    )

    # ------------------------------------------------------------
    # NASA NIGHTTIME LIGHTING
    # ------------------------------------------------------------
    try:
        lighting = calculate_route_lighting(point)
    except Exception as error:
        print("Live lighting error:", error)
        lighting = fallbacks["lighting"]

    lighting_score = float(lighting.get("lighting_score", 0) or 0)

    # ------------------------------------------------------------
    # TOMTOM TRAFFIC
    # ------------------------------------------------------------
    try:
        traffic = calculate_route_traffic(point)
    except Exception as error:
        print("Live traffic error:", error)
        traffic = fallbacks["traffic"]

    traffic_score = float(traffic.get("traffic_score", 0) or 0)

    # ------------------------------------------------------------
    # OPEN-METEO WEATHER
    # ------------------------------------------------------------
    try:
        weather = calculate_route_weather(point)
    except Exception as error:
        print("Live weather error:", error)
        weather = fallbacks["weather"]

    weather_score = float(weather.get("weather_score", 0) or 0)
    rainfall_mm = float(weather.get("average_precipitation_mm", 0) or 0)

    # ------------------------------------------------------------
    # OPEN-METEO VISIBILITY
    # ------------------------------------------------------------
    try:
        visibility = calculate_route_visibility(point)
        visibility_km = float(visibility.get("visibility_km", 0) or 0)
    except Exception as error:
        print("Live visibility error:", error)
        visibility_km = 0.0

    # ------------------------------------------------------------
    # CRIME AT CURRENT AREA
    # ------------------------------------------------------------
    try:
        crime = calculate_route_crime(point)
    except Exception as error:
        print("Live crime error:", error)
        crime = fallbacks["crime"]

    v1_crime = get_v1_crime_features(crime)

    # ------------------------------------------------------------
    # EMERGENCY ACCESS BASELINE
    # ------------------------------------------------------------
    emergency = get_emergency_features(crime)

    # ------------------------------------------------------------
    # SAFE ROUTE V1 DEEP-LEARNING PREDICTION
    # ------------------------------------------------------------
    try:
        v1_prediction = predict_route_safety(
            traffic_index=traffic_score,
            lighting_score=lighting_score,
            rainfall_mm=rainfall_mm,
            visibility_km=visibility_km,
            violent_crime_count=v1_crime["violent_crime_count"],
            property_crime_count=v1_crime["property_crime_count"],
            women_crime_count=v1_crime["women_crime_count"],
            other_crime_count=v1_crime["other_crime_count"],
            total_crime_count=v1_crime["total_crime_count"],
            crime_rate_per_1000_exposure=v1_crime["crime_rate_per_1000_exposure"],
            nearest_police_km=emergency["nearest_police_km"],
            nearest_hospital_km=emergency["nearest_hospital_km"],
            emergency_access_score=emergency["emergency_access_score"],
            month=datetime.now().month
        )

        safety_score = float(v1_prediction.get("safety_score", 0) or 0)
        risk_level = v1_prediction.get("risk_level", "Unknown")
    except Exception as error:
        print("Live V1 prediction error:", error)
        safety_score = 0.0
        risk_level = "Unavailable"

    crime_risk = float(crime.get("crime_risk_score", 0) or 0)
    crime_safety = float(crime.get("crime_safety_score", 0) or 0)

    # A clear alert is easier to understand on a phone.
    if crime_risk >= 60:
        safety_alert = "High crime risk detected in this area."
        alert_type = "crime"
    elif safety_score < 45:
        safety_alert = "Higher safety risk detected in this area."
        alert_type = "safety"
    elif weather_score < 60:
        safety_alert = "Weather conditions may require extra caution."
        alert_type = "weather"
    elif lighting_score < 40:
        safety_alert = "Low nighttime lighting detected in this area."
        alert_type = "lighting"
    elif traffic_score < 40:
        safety_alert = "Heavy traffic conditions detected in this area."
        alert_type = "traffic"
    else:
        safety_alert = "No major safety warning detected right now."
        alert_type = "none"

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "safety_score": round(safety_score, 2),
        "risk_level": risk_level,
        "safety_alert": safety_alert,
        "alert_type": alert_type,
        "lighting": {
            "score": round(lighting_score, 2),
            "value": lighting.get("lighting_value", 0)
        },
        "traffic": {
            "score": round(traffic_score, 2),
            "current_speed_kmph": traffic.get("average_current_speed_kmph", 0),
            "free_flow_speed_kmph": traffic.get("average_free_flow_speed_kmph", 0),
            "delay_seconds": traffic.get("average_delay_seconds", 0),
            "confidence": traffic.get("average_confidence", 0)
        },
        "weather": {
            "score": round(weather_score, 2),
            "condition": weather.get("weather_condition", "Unavailable"),
            "temperature_c": weather.get("average_temperature_c", 0),
            "humidity_percent": weather.get("average_humidity_percent", 0),
            "precipitation_mm": weather.get("average_precipitation_mm", 0),
            "rain_mm": weather.get("average_rain_mm", 0),
            "wind_speed_kmph": weather.get("average_wind_speed_kmph", 0),
            "wind_gusts_kmph": weather.get("average_wind_gusts_kmph", 0),
            "cloud_cover_percent": weather.get("average_cloud_cover_percent", 0),
            "visibility_km": round(visibility_km, 2)
        },
        "crime": {
            "risk_score": round(crime_risk, 2),
            "safety_score": round(crime_safety, 2),
            "location": crime.get("crime_location", "Unknown"),
            "location_type": crime.get("crime_location_type", "unknown"),
            "data_type": crime.get("crime_data_type", "unavailable"),
            "violent_count": v1_crime["violent_crime_count"],
            "property_count": v1_crime["property_crime_count"],
            "women_count": v1_crime["women_crime_count"],
            "total_count": v1_crime["total_crime_count"]
        },
        "emergency": {
            "nearest_police_km": emergency["nearest_police_km"],
            "nearest_hospital_km": emergency["nearest_hospital_km"],
            "access_score": emergency["emergency_access_score"]
        },
        "model": "SafeRoute V1 Multi-Branch Deep Learning"
    }
