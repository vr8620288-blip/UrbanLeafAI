import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3"

CACHE_DIR = Path("data/real_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AIR_QUALITY_FILE = CACHE_DIR / "air_quality.csv"


def get_headers():
    if not OPENAQ_API_KEY:
        raise ValueError(
            "OPENAQ_API_KEY not found. Check your .env file."
        )

    return {
        "X-API-Key": OPENAQ_API_KEY
    }


def get_nearby_locations(
    latitude,
    longitude,
    radius=25000
):
    """
    Find OpenAQ monitoring locations near a coordinate.
    """

    url = f"{BASE_URL}/locations"

    params = {
        "coordinates": f"{latitude},{longitude}",
        "radius": radius,
        "limit": 100
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json().get("results", [])


def get_location_details(location_id):
    """
    Get complete metadata for an OpenAQ location.

    This includes sensors and the pollutants they measure.
    """

    url = f"{BASE_URL}/locations/{location_id}"

    response = requests.get(
        url,
        headers=get_headers(),
        timeout=20
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    return results[0]


def get_location_latest(location_id):
    """
    Get latest measurements for a location.
    """

    url = f"{BASE_URL}/locations/{location_id}/latest"

    params = {
        "limit": 100
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json().get("results", [])


def build_sensor_map(location_details):
    """
    Create:

    sensor ID
        ↓
    parameter + units
    """

    sensor_map = {}

    sensors = location_details.get("sensors", [])

    for sensor in sensors:

        sensor_id = sensor.get("id")

        parameter = sensor.get("parameter", {})

        sensor_map[sensor_id] = {
            "parameter": parameter.get(
                "name",
                "unknown"
            ),
            "units": parameter.get(
                "units",
                ""
            ),
            "display_name": parameter.get(
                "displayName",
                parameter.get(
                    "name",
                    "Unknown"
                )
            )
        }

    return sensor_map


def extract_location_data(location):
    """
    Extract measurements and attach the correct
    pollutant name using sensor metadata.
    """

    location_id = location.get("id")

    station = location.get(
        "name",
        "Unknown"
    )

    coordinates = location.get(
        "coordinates"
    ) or {}

    latitude = coordinates.get(
        "latitude"
    )

    longitude = coordinates.get(
        "longitude"
    )

    # Get sensor metadata
    details = get_location_details(
        location_id
    )

    if not details:
        return []

    sensor_map = build_sensor_map(
        details
    )

    # Get latest values
    measurements = get_location_latest(
        location_id
    )

    rows = []

    for measurement in measurements:

        sensor_id = measurement.get(
            "sensorsId"
        )

        value = measurement.get(
            "value"
        )

        if value is None:
            continue

        sensor_info = sensor_map.get(
            sensor_id,
            {}
        )

        parameter = sensor_info.get(
            "parameter",
            "unknown"
        )

        units = sensor_info.get(
            "units",
            ""
        )

        display_name = sensor_info.get(
            "display_name",
            parameter
        )

        datetime_data = measurement.get(
            "datetime",
            {}
        )

        if isinstance(
            datetime_data,
            dict
        ):
            local_datetime = datetime_data.get(
                "local"
            )
        else:
            local_datetime = None

        rows.append({

            "location_id": location_id,

            "station": station,

            "latitude": latitude,

            "longitude": longitude,

            "sensor_id": sensor_id,

            "parameter": parameter,

            "display_name": display_name,

            "units": units,

            "value": value,

            "datetime": local_datetime
        })

    return rows


def fetch_pune_air_quality():
    """
    Download real Pune air-quality data
    from OpenAQ.

    Pune center:
    18.5204, 73.8567

    Search radius:
    25 km
    """

    pune_lat = 18.5204
    pune_lon = 73.8567

    locations = get_nearby_locations(
        pune_lat,
        pune_lon,
        radius=25000
    )

    print(
        f"OpenAQ stations found: "
        f"{len(locations)}"
    )

    all_rows = []

    for index, location in enumerate(
        locations,
        start=1
    ):

        print(
            f"Processing station "
            f"{index}/{len(locations)}: "
            f"{location.get('name')}"
        )

        try:

            rows = extract_location_data(
                location
            )

            all_rows.extend(rows)

        except Exception as e:

            print(
                f"Skipped station "
                f"{location.get('name')}: "
                f"{e}"
            )

    if not all_rows:

        print(
            "No OpenAQ measurements found."
        )

        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows
    )

    # Remove duplicate records
    df = df.drop_duplicates()

    # Save locally
    df.to_csv(
        AIR_QUALITY_FILE,
        index=False
    )

    print(
        f"\nSaved {len(df)} "
        f"measurements to "
        f"{AIR_QUALITY_FILE}"
    )

    print(
        "\nParameters found:"
    )

    print(
        df[
            [
                "parameter",
                "display_name",
                "units"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    return df


def load_cached_air_quality():
    """
    Load previously downloaded
    real air-quality data.
    """

    if not AIR_QUALITY_FILE.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            AIR_QUALITY_FILE
        )

    except Exception as e:

        print(
            f"Could not read cached "
            f"air-quality data: {e}"
        )

        return pd.DataFrame()