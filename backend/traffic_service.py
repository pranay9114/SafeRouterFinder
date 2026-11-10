import os
import requests
from dotenv import load_dotenv
from statistics import mean

load_dotenv()

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

TOMTOM_TRAFFIC_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)


def get_traffic_data(latitude, longitude):
    """
    Get real-time traffic information for the
    road segment closest to the coordinate.
    """

    if not TOMTOM_API_KEY:
        raise ValueError("TOMTOM_API_KEY not found in .env")

    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{latitude},{longitude}",
        "unit": "kmph"
    }

    response = requests.get(
        TOMTOM_TRAFFIC_URL,
        params=params,
        timeout=15
    )

    if response.status_code != 200:
        raise Exception(
            f"TomTom API error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    flow = data.get("flowSegmentData")

    if not flow:
        raise Exception("TomTom returned no flowSegmentData")

    current_speed = flow.get("currentSpeed")
    free_flow_speed = flow.get("freeFlowSpeed")
    current_travel_time = flow.get("currentTravelTime")
    free_flow_travel_time = flow.get("freeFlowTravelTime")
    confidence = flow.get("confidence")
    road_closure = flow.get("roadClosure", False)

    if free_flow_speed and free_flow_speed > 0:
        speed_ratio = current_speed / free_flow_speed
    else:
        speed_ratio = None

    if (
        current_travel_time is not None
        and free_flow_travel_time is not None
    ):
        delay_seconds = (
            current_travel_time -
            free_flow_travel_time
        )
    else:
        delay_seconds = None

    return {
        "current_speed_kmph": current_speed,
        "free_flow_speed_kmph": free_flow_speed,
        "current_travel_time_seconds": current_travel_time,
        "free_flow_travel_time_seconds": free_flow_travel_time,
        "traffic_delay_seconds": delay_seconds,
        "speed_ratio": speed_ratio,
        "confidence": confidence,
        "road_closure": road_closure
    }


def sample_route_points(coordinates, sample_count=8):
    """
    Select evenly distributed points from a GraphHopper route.

    GraphHopper coordinates are:
        [longitude, latitude]
    """

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

    return [coordinates[i] for i in indexes]


def calculate_route_traffic(coordinates):
    """
    Calculate aggregated real-time traffic features
    for an entire route.
    """

    sample_points = sample_route_points(
        coordinates,
        sample_count=8
    )

    results = []

    for point in sample_points:

        longitude = point[0]
        latitude = point[1]

        try:
            traffic = get_traffic_data(
                latitude,
                longitude
            )

            results.append(traffic)

        except Exception as e:
            print(
                f"TomTom traffic request failed "
                f"at {latitude}, {longitude}: {e}"
            )

    if not results:

        return {
            "traffic_score": 0,
            "average_current_speed_kmph": 0,
            "average_free_flow_speed_kmph": 0,
            "average_delay_seconds": 0,
            "average_speed_ratio": 0,
            "average_confidence": 0,
            "road_closure_count": 0,
            "traffic_sample_count": 0
        }

    speed_ratios = [
        r["speed_ratio"]
        for r in results
        if r["speed_ratio"] is not None
    ]

    current_speeds = [
        r["current_speed_kmph"]
        for r in results
        if r["current_speed_kmph"] is not None
    ]

    free_flow_speeds = [
        r["free_flow_speed_kmph"]
        for r in results
        if r["free_flow_speed_kmph"] is not None
    ]

    delays = [
        r["traffic_delay_seconds"]
        for r in results
        if r["traffic_delay_seconds"] is not None
    ]

    confidences = [
        r["confidence"]
        for r in results
        if r["confidence"] is not None
    ]

    road_closures = sum(
        1
        for r in results
        if r["road_closure"]
    )

    average_speed_ratio = (
        mean(speed_ratios)
        if speed_ratios
        else 0
    )

    traffic_score = max(
        0,
        min(
            100,
            round(average_speed_ratio * 100)
        )
    )

    return {
        "traffic_score": traffic_score,

        "average_current_speed_kmph": round(
            mean(current_speeds), 2
        ) if current_speeds else 0,

        "average_free_flow_speed_kmph": round(
            mean(free_flow_speeds), 2
        ) if free_flow_speeds else 0,

        "average_delay_seconds": round(
            mean(delays), 2
        ) if delays else 0,

        "average_speed_ratio": round(
            average_speed_ratio, 3
        ),

        "average_confidence": round(
            mean(confidences), 3
        ) if confidences else 0,

        "road_closure_count": road_closures,

        "traffic_sample_count": len(results)
    }


if __name__ == "__main__":

    latitude = 14.4673
    longitude = 78.8242

    print("Testing TomTom real-time traffic...")
    print(f"Location: {latitude}, {longitude}")
    print()

    try:

        traffic = get_traffic_data(
            latitude,
            longitude
        )

        print("TomTom Traffic Result")
        print("---------------------")

        for key, value in traffic.items():
            print(f"{key}: {value}")

    except Exception as e:

        print("Traffic request failed:")
        print(e)