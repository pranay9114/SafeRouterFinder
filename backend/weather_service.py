
import requests
from statistics import mean


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# GET WEATHER FOR ONE LOCATION
# ============================================================

def get_weather_data(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_gusts_10m"
        ),

        "hourly": "visibility",

        "forecast_days": 1,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "auto"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})

    return {
        "temperature_c": current.get("temperature_2m", 0),
        "humidity_percent": current.get(
            "relative_humidity_2m", 0
        ),
        "precipitation_mm": current.get(
            "precipitation", 0
        ),
        "rain_mm": current.get("rain", 0),
        "weather_code": current.get(
            "weather_code", 0
        ),
        "cloud_cover_percent": current.get(
            "cloud_cover", 0
        ),
        "wind_speed_kmph": current.get(
            "wind_speed_10m", 0
        ),
        "wind_gusts_kmph": current.get(
            "wind_gusts_10m", 0
        ),
        "timezone": data.get(
            "timezone", "Asia/Kolkata"
        )
    }


# ============================================================
# WEATHER CONDITION
# ============================================================

def get_weather_condition(weather_code):

    if weather_code == 0:
        return "Clear"

    if weather_code in [1, 2]:
        return "Partly Cloudy"

    if weather_code == 3:
        return "Cloudy"

    if weather_code in [45, 48]:
        return "Foggy"

    if weather_code in [51, 53, 55, 56, 57]:
        return "Drizzle"

    if weather_code in [61, 63, 65, 66, 67]:
        return "Rain"

    if weather_code in [71, 73, 75, 77]:
        return "Snow"

    if weather_code in [80, 81, 82]:
        return "Rain Showers"

    if weather_code in [85, 86]:
        return "Snow Showers"

    if weather_code in [95, 96, 99]:
        return "Thunderstorm"

    return "Unknown"


# ============================================================
# WEATHER RISK SCORE
# ============================================================

def get_weather_risk_score(weather):

    score = 100.0

    rain = weather["rain_mm"]
    precipitation = weather["precipitation_mm"]
    wind = weather["wind_speed_kmph"]
    gusts = weather["wind_gusts_kmph"]
    cloud = weather["cloud_cover_percent"]

    # Rain penalty
    if rain > 0:
        score -= min(rain * 10, 30)

    # Precipitation penalty
    if precipitation > 0:
        score -= min(precipitation * 5, 20)

    # Strong wind penalty
    if wind > 30:
        score -= 10

    if wind > 50:
        score -= 15

    # Strong wind gust penalty
    if gusts > 50:
        score -= 10

    if gusts > 70:
        score -= 15

    # Very cloudy conditions
    if cloud > 90:
        score -= 5

    return round(
        max(0, min(100, score)),
        2
    )


# ============================================================
# SAMPLE ROUTE POINTS
# ============================================================

def sample_route_points(
    coordinates,
    max_samples=5
):

    if not coordinates:
        return []

    total = len(coordinates)

    if total <= max_samples:
        return coordinates

    indices = []

    for i in range(max_samples):
        index = int(
            i * (total - 1) /
            (max_samples - 1)
        )

        indices.append(index)

    return [
        coordinates[index]
        for index in indices
    ]


# ============================================================
# CALCULATE WEATHER FOR ENTIRE ROUTE
# ============================================================

def calculate_route_weather(coordinates):

    sampled_points = sample_route_points(
        coordinates,
        max_samples=5
    )

    if not sampled_points:

        return {
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
        }

    weather_results = []

    for coordinate in sampled_points:

        # GraphHopper coordinates are:
        # [longitude, latitude]

        longitude = coordinate[0]
        latitude = coordinate[1]

        try:

            weather = get_weather_data(
                latitude,
                longitude
            )

            weather["weather_score"] = (
                get_weather_risk_score(weather)
            )

            weather["weather_condition"] = (
                get_weather_condition(
                    weather["weather_code"]
                )
            )

            weather_results.append(weather)

        except Exception as error:

            print(
                "Weather API error:",
                error
            )

    if not weather_results:

        return {
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
        }

    return {

        "weather_score": round(
            mean(
                x["weather_score"]
                for x in weather_results
            ),
            2
        ),

        "weather_condition":
            weather_results[0][
                "weather_condition"
            ],

        "average_temperature_c": round(
            mean(
                x["temperature_c"]
                for x in weather_results
            ),
            2
        ),

        "average_humidity_percent": round(
            mean(
                x["humidity_percent"]
                for x in weather_results
            ),
            2
        ),

        "average_precipitation_mm": round(
            mean(
                x["precipitation_mm"]
                for x in weather_results
            ),
            2
        ),

        "average_rain_mm": round(
            mean(
                x["rain_mm"]
                for x in weather_results
            ),
            2
        ),

        "average_wind_speed_kmph": round(
            mean(
                x["wind_speed_kmph"]
                for x in weather_results
            ),
            2
        ),

        "average_wind_gusts_kmph": round(
            mean(
                x["wind_gusts_kmph"]
                for x in weather_results
            ),
            2
        ),

        "average_cloud_cover_percent": round(
            mean(
                x["cloud_cover_percent"]
                for x in weather_results
            ),
            2
        ),

        "weather_sample_count":
            len(weather_results)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_coordinates = [
        [78.8241, 14.4674],
        [78.8300, 14.4700],
        [78.8400, 14.4800]
    ]

    result = calculate_route_weather(
        test_coordinates
    )

    print("\nWeather Test Result")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")