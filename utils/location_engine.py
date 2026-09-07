from math import radians, sin, cos, sqrt, atan2


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates.
    """

    R = 6371.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def find_nearest_zone(latitude, longitude, zones_df):
    """
    Find the UrbanLeaf zone closest to the user's GPS location.
    """

    df = zones_df.copy()

    latitude_column = None
    longitude_column = None

    for col in ["latitude", "lat", "Latitude", "LATITUDE"]:
        if col in df.columns:
            latitude_column = col
            break

    for col in [
        "longitude",
        "lon",
        "lng",
        "Longitude",
        "LONGITUDE",
    ]:
        if col in df.columns:
            longitude_column = col
            break

    if latitude_column is None or longitude_column is None:
        raise ValueError(
            "Dataset must contain latitude and longitude columns."
        )

    df = df.dropna(
        subset=[latitude_column, longitude_column]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid zone coordinates found."
        )

    df["_distance_km"] = df.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            row[latitude_column],
            row[longitude_column],
        ),
        axis=1,
    )

    nearest_index = df["_distance_km"].idxmin()

    nearest_zone = df.loc[nearest_index].copy()

    distance_km = float(
        nearest_zone["_distance_km"]
    )

    nearest_zone = nearest_zone.drop(
        labels=["_distance_km"]
    )

    return nearest_zone, distance_km


def get_zone_name(zone):
    """
    Safely retrieve zone name.
    """

    for key in [
        "zone",
        "Zone",
        "zone_name",
        "Zone Name",
        "name",
    ]:
        if key in zone.index:
            return str(zone[key])

    return "Nearest UrbanLeaf Zone"


def get_priority_level(zone):
    """
    Safely retrieve priority level.
    """

    for key in [
        "priority_level",
        "Priority Level",
        "priority",
        "Priority",
        "level",
    ]:
        if key in zone.index:
            return str(zone[key]).upper()

    return "HIGH"


def get_priority_score(zone):
    """
    Safely retrieve priority score.
    """

    for key in [
        "priority_score",
        "Priority Score",
        "score",
        "Score",
    ]:
        if key in zone.index:
            try:
                return float(zone[key])
            except Exception:
                pass

    return None