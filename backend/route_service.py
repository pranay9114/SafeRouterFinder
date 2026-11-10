import requests
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


# Allow React frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAPHOPPER_URL = "http://localhost:8989/route"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "SafeRouteFinder/1.0"
}


# ============================================================
# REQUEST MODEL
# ============================================================

class RouteRequest(BaseModel):
    source: str
    destination: str


# ============================================================
# GEOCODE PLACE NAME
# PLACE NAME -> LATITUDE/LONGITUDE
# ============================================================

def geocode_location(location_name):

    params = {
        "q": f"{location_name}, Andhra Pradesh, India",
        "format": "json",
        "limit": 1
    }

    try:

        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        return {
            "name": data[0]["display_name"],
            "lat": float(data[0]["lat"]),
            "lon": float(data[0]["lon"])
        }

    except Exception as error:

        print("Geocoding error:", error)

        return None


# ============================================================
# GET ROUTE FROM GRAPHOPPER
# ============================================================

def get_route(points):

    params = [
        ("profile", "car"),
        ("points_encoded", "false")
    ]

    for lat, lon in points:

        params.append(
            ("point", f"{lat},{lon}")
        )

    try:

        response = requests.get(
            GRAPHOPPER_URL,
            params=params,
            timeout=120
        )

        if response.status_code != 200:

            print(
                "GraphHopper error:",
                response.text
            )

            return None

        data = response.json()

        if "paths" not in data or len(data["paths"]) == 0:

            return None

        return data["paths"][0]

    except Exception as error:

        print("GraphHopper connection error:", error)

        return None


# ============================================================
# CHECK IF TWO ROUTES ARE TOO SIMILAR
# ============================================================

def is_similar_route(route1, route2):

    distance_difference = abs(
        route1["distance"] - route2["distance"]
    ) / 1000

    time_difference = abs(
        route1["time"] - route2["time"]
    ) / 60000

    # Consider routes similar if both
    # distance and time are very close
    if distance_difference < 2 and time_difference < 5:

        return True

    return False


# ============================================================
# CREATE AUTOMATIC CANDIDATE DETOUR POINTS
# ============================================================

def create_candidate_points(route):

    coordinates = route["points"]["coordinates"]

    candidates = []

    total = len(coordinates)

    positions = [
        0.25,
        0.40,
        0.55,
        0.70,
        0.80
    ]

    for position in positions:

        index = int(total * position)

        # GraphHopper coordinates:
        # [longitude, latitude]
        lon, lat = coordinates[index]

        offsets = [
            (0.03, 0.03),
            (-0.03, 0.03),
            (0.03, -0.03),
            (-0.03, -0.03)
        ]

        for lat_offset, lon_offset in offsets:

            candidate_lat = lat + lat_offset

            candidate_lon = lon + lon_offset

            candidates.append(
                (
                    candidate_lat,
                    candidate_lon
                )
            )

    return candidates


# ============================================================
# CONVERT ROUTES TO FRONTEND FORMAT
# ============================================================

def format_routes(routes):

    output = []

    for i, route in enumerate(routes, start=1):

        route_info = {

            "route_id": i,

            "distance_km": round(
                route["distance"] / 1000,
                2
            ),

            "time_minutes": round(
                route["time"] / 60000,
                2
            ),

            # GraphHopper coordinates:
            # [longitude, latitude]
            "coordinates": route["points"]["coordinates"]
        }

        output.append(route_info)

    return output


# ============================================================
# HOME API
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Safe Route Finder Backend is Running"
    }


# ============================================================
# FIND ROUTES API
# ============================================================

@app.post("/routes")
def find_routes(request: RouteRequest):

    source_name = request.source.strip()

    destination_name = request.destination.strip()


    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not source_name or not destination_name:

        return {
            "success": False,
            "message": "Please enter both source and destination."
        }


    print("\n" + "=" * 60)

    print("SOURCE:", source_name)

    print("DESTINATION:", destination_name)

    print("=" * 60)


    # --------------------------------------------------------
    # GEOCODE SOURCE
    # --------------------------------------------------------

    print("\n📍 Finding source location...")

    source_location = geocode_location(source_name)

    if source_location is None:

        return {
            "success": False,
            "message": f"Could not find source location: {source_name}"
        }


    # --------------------------------------------------------
    # GEOCODE DESTINATION
    # --------------------------------------------------------

    print("\n🏁 Finding destination location...")

    destination_location = geocode_location(destination_name)

    if destination_location is None:

        return {
            "success": False,
            "message": (
                f"Could not find destination location: "
                f"{destination_name}"
            )
        }


    # --------------------------------------------------------
    # CREATE COORDINATES
    # --------------------------------------------------------

    source = (
        source_location["lat"],
        source_location["lon"]
    )

    destination = (
        destination_location["lat"],
        destination_location["lon"]
    )


    print("\nSource coordinates:", source)

    print("Destination coordinates:", destination)


    # --------------------------------------------------------
    # STEP 1 — GET MAIN ROUTE
    # --------------------------------------------------------

    print("\n🔍 Finding main route...")

    main_route = get_route([
        source,
        destination
    ])


    if main_route is None:

        return {
            "success": False,
            "message": (
                "Could not find a route. "
                "Make sure GraphHopper is running and "
                "the locations are inside your OSM map area."
            )
        }


    routes = [main_route]

    print("✅ Main route found.")


    # --------------------------------------------------------
    # STEP 2 — CREATE CANDIDATE POINTS
    # --------------------------------------------------------

    print("\n🔍 Generating alternative routes...")

    candidate_points = create_candidate_points(
        main_route
    )


    # --------------------------------------------------------
    # STEP 3 — FIND DISTINCT ROUTES
    # --------------------------------------------------------

    for candidate in candidate_points:

        # Maximum 3 routes
        if len(routes) >= 3:
            break


        candidate_route = get_route([
            source,
            candidate,
            destination
        ])


        if candidate_route is None:
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

            routes.append(candidate_route)

            print(
                f"✅ Alternative route found: "
                f"{candidate_route['distance'] / 1000:.2f} km"
            )


    # --------------------------------------------------------
    # FORMAT RESULTS
    # --------------------------------------------------------

    formatted_routes = format_routes(routes)


    print("\n" + "#" * 60)

    print("ROUTING COMPLETED")

    print(f"Number of routes: {len(formatted_routes)}")

    print("#" * 60)


    # --------------------------------------------------------
    # RETURN TO REACT
    # --------------------------------------------------------

    return {

        "success": True,

        "message": f"{len(formatted_routes)} routes found",

        "source": source_location,

        "destination": destination_location,

        "routes": formatted_routes
    }