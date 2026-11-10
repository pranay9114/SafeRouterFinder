import os
import glob

import h5py
import numpy as np


# ============================================================
# NASA NIGHTTIME LIGHT DATA
# ============================================================

DATASET_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "datasets",
    "nighttime_lights"
)

LIGHTING_DATASET = (
    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/"
    "Data Fields/"
    "Gap_Filled_DNB_BRDF-Corrected_NTL"
)

LAT_DATASET = (
    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/"
    "Data Fields/lat"
)

LON_DATASET = (
    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/"
    "Data Fields/lon"
)


# ============================================================
# LOAD NASA FILES
# ============================================================

def load_nighttime_light_data():

    files = glob.glob(
        os.path.join(
            DATASET_FOLDER,
            "*.h5"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No NASA .h5 files found in: "
            + DATASET_FOLDER
        )

    datasets = []

    print("\nLoading NASA nighttime-light data...")

    for file_path in files:

        print(
            "Loading:",
            os.path.basename(file_path)
        )

        with h5py.File(
            file_path,
            "r"
        ) as h5_file:

            light = np.array(
                h5_file[LIGHTING_DATASET][:]
            )

            lat = np.array(
                h5_file[LAT_DATASET][:]
            )

            lon = np.array(
                h5_file[LON_DATASET][:]
            )

        datasets.append({
            "file": os.path.basename(file_path),
            "light": light,
            "lat": lat,
            "lon": lon
        })

    print(
        f"Loaded {len(datasets)} NASA tile(s)."
    )

    return datasets


# ============================================================
# FIND NEAREST ARRAY INDEX
# ============================================================

def nearest_index(values, target):

    values = np.asarray(values)

    # Latitude/longitude arrays are normally monotonic.
    ascending = values[0] < values[-1]

    if not ascending:
        reversed_values = values[::-1]

        index = np.searchsorted(
            reversed_values,
            target
        )

        index = min(
            index,
            len(reversed_values) - 1
        )

        candidates = [
            index,
            max(index - 1, 0)
        ]

        best = min(
            candidates,
            key=lambda i:
                abs(
                    reversed_values[i]
                    - target
                )
        )

        return len(values) - 1 - best

    index = np.searchsorted(
        values,
        target
    )

    if index == 0:
        return 0

    if index >= len(values):
        return len(values) - 1

    before = index - 1
    after = index

    if abs(values[before] - target) <= abs(
        values[after] - target
    ):
        return before

    return after


# ============================================================
# GET LIGHT VALUE FOR ONE LOCATION
# ============================================================

def get_light_value(
    latitude,
    longitude,
    datasets
):

    best_value = None

    for dataset in datasets:

        lat_array = dataset["lat"]
        lon_array = dataset["lon"]
        light_array = dataset["light"]

        # Check whether this tile contains
        # the requested coordinate.

        if (
            latitude < lat_array.min()
            or latitude > lat_array.max()
            or longitude < lon_array.min()
            or longitude > lon_array.max()
        ):
            continue

        row = nearest_index(
            lat_array,
            latitude
        )

        column = nearest_index(
            lon_array,
            longitude
        )

        value = float(
            light_array[row, column]
        )

        # Ignore invalid NASA fill values.
        if not np.isfinite(value):
            continue

        if value < 0:
            continue

        best_value = value
        break

    return best_value


# ============================================================
# SAMPLE ROUTE
# ============================================================

def sample_route(
    route_coordinates,
    max_samples=300
):

    if not route_coordinates:
        return []

    coordinates = route_coordinates

    if len(coordinates) <= max_samples:
        return coordinates

    indexes = np.linspace(
        0,
        len(coordinates) - 1,
        max_samples,
        dtype=int
    )

    return [
        coordinates[index]
        for index in indexes
    ]


# ============================================================
# CALCULATE ROUTE NIGHTTIME LIGHT
# ============================================================

def calculate_route_lighting(
    route_coordinates
):

    datasets = load_nighttime_light_data()

    sampled_coordinates = sample_route(
        route_coordinates
    )

    light_values = []

    for coordinate in sampled_coordinates:

        # GraphHopper gives:
        # [longitude, latitude]

        longitude = coordinate[0]
        latitude = coordinate[1]

        value = get_light_value(
            latitude,
            longitude,
            datasets
        )

        if value is not None:
            light_values.append(value)

    if not light_values:

        return {
            "lighting_value": 0,
            "lighting_score": 0,
            "sample_count": 0
        }

    light_values = np.array(
        light_values,
        dtype=float
    )

    # Median is used because nighttime-light
    # values can contain very bright locations
    # that would distort a simple average.

    median_light = float(
        np.median(light_values)
    )

    # Use the 95th percentile of the available
    # satellite values as a practical bright reference.

    all_valid_values = []

    for dataset in datasets:

        values = dataset["light"]

        valid_values = values[
            np.isfinite(values)
            & (values >= 0)
        ]

        # Randomly sample large datasets so that
        # processing remains efficient.

        if len(valid_values) > 100000:

            indexes = np.linspace(
                0,
                len(valid_values) - 1,
                100000,
                dtype=int
            )

            valid_values = valid_values[
                indexes
            ]

        all_valid_values.extend(
            valid_values.tolist()
        )

    if not all_valid_values:

        lighting_score = 0

    else:

        reference_value = float(
            np.percentile(
                all_valid_values,
                95
            )
        )

        if reference_value <= 0:

            lighting_score = 0

        else:

            # Log scaling prevents extremely bright
            # urban pixels from dominating the score.

            score = (
                np.log1p(median_light)
                /
                np.log1p(reference_value)
            ) * 100

            lighting_score = int(
                np.clip(
                    score,
                    0,
                    100
                )
            )

    return {
        "lighting_value": round(
            median_light,
            4
        ),

        "lighting_score": lighting_score,

        "sample_count": len(
            light_values
        )
    }


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SAFE ROUTE FINDER")
    print("NASA NIGHTTIME LIGHT TEST")
    print("=" * 60)

    # Example coordinate near Andhra Pradesh.
    # Format: [longitude, latitude]

    test_route = [
        [78.8242, 14.4673],
        [78.90, 14.50],
        [79.00, 14.60],
        [79.20, 14.80],
        [79.40, 15.00]
    ]

    result = calculate_route_lighting(
        test_route
    )

    print()
    print("Nighttime light value:")
    print(
        result["lighting_value"]
    )

    print()
    print("Lighting score:")
    print(
        result["lighting_score"],
        "/ 100"
    )

    print()
    print("Samples used:")
    print(
        result["sample_count"]
    )

    print()
    print("=" * 60)
    print("LIGHTING TEST COMPLETE")
    print("=" * 60)