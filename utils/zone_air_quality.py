import math
from pathlib import Path

import pandas as pd


ZONE_FILE = Path("data/urban_zones.csv")
AQ_FILE = Path("data/real_data/air_quality.csv")
OUTPUT_FILE = Path("data/real_data/zone_air_quality.csv")


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two latitude/longitude points."""

    R = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return R * 2 * math.asin(math.sqrt(a))


def load_data():
    """Load UrbanLeaf zones and real OpenAQ data."""

    zones = pd.read_csv(ZONE_FILE)
    aq = pd.read_csv(AQ_FILE)

    aq["datetime"] = pd.to_datetime(
        aq["datetime"],
        errors="coerce"
    )

    aq["value"] = pd.to_numeric(
        aq["value"],
        errors="coerce"
    )

    aq = aq.dropna(
        subset=[
            "latitude",
            "longitude",
            "value"
        ]
    )

    return zones, aq


def get_latest_measurements(aq):
    """
    Keep the latest measurement for every
    station + pollutant combination.
    """

    if "parameter" not in aq.columns:
        raise ValueError(
            "The air_quality.csv file does not contain "
            "a 'parameter' column. Run the updated "
            "real_data.py first."
        )

    aq = aq.sort_values(
        "datetime"
    )

    latest = (
        aq
        .groupby(
            [
                "location_id",
                "parameter"
            ],
            as_index=False
        )
        .tail(1)
    )

    return latest


def choose_best_station(zone_lat, zone_lon, station_data):
    """
    Find the closest monitoring station.

    Prefer a station that has PM2.5 if available.
    """

    stations = (
        station_data[
            [
                "location_id",
                "station",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    stations["distance_km"] = stations.apply(
        lambda row: haversine_km(
            zone_lat,
            zone_lon,
            row["latitude"],
            row["longitude"]
        ),
        axis=1
    )

    # Prefer nearest station overall.
    stations = stations.sort_values(
        "distance_km"
    )

    return stations.iloc[0]


def match_zones_to_air_quality():
    """Match every UrbanLeaf zone to its nearest real station."""

    zones, aq = load_data()

    latest = get_latest_measurements(aq)

    results = []

    for _, zone in zones.iterrows():

        zone_name = zone["zone"]
        zone_lat = zone["latitude"]
        zone_lon = zone["longitude"]

        station = choose_best_station(
            zone_lat,
            zone_lon,
            latest
        )

        station_id = station["location_id"]

        station_measurements = latest[
            latest["location_id"] == station_id
        ].copy()

        result = {
            "zone": zone_name,
            "zone_latitude": zone_lat,
            "zone_longitude": zone_lon,

            "nearest_station": station["station"],

            "station_latitude": station["latitude"],
            "station_longitude": station["longitude"],

            "distance_km": round(
                station["distance_km"],
                2
            )
        }

        # Add pollutant values dynamically.
        for _, measurement in station_measurements.iterrows():

            parameter = str(
                measurement["parameter"]
            ).lower()

            value = measurement["value"]
            units = measurement.get(
                "units",
                ""
            )

            if parameter == "pm25":
                result["pm25"] = value
                result["pm25_units"] = units

            elif parameter == "pm10":
                result["pm10"] = value
                result["pm10_units"] = units

            elif parameter == "no2":
                result["no2"] = value
                result["no2_units"] = units

            elif parameter == "so2":
                result["so2"] = value
                result["so2_units"] = units

            elif parameter == "co":
                result["co"] = value
                result["co_units"] = units

            elif parameter == "o3":
                result["o3"] = value
                result["o3_units"] = units

        results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return result_df


if __name__ == "__main__":

    print("\n🌍 URBANLEAF REAL AIR QUALITY MATCHING")
    print("=" * 50)

    df = match_zones_to_air_quality()

    print(
        f"\nMatched {len(df)} zones."
    )

    print("\nResults:\n")

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )