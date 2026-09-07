import pandas as pd


def calculate_priority_score(row):
    """
    Calculate urban greening priority score from 0 to 100.
    Higher score = higher priority.
    """

    # Heat score
    heat_score = min(
        max((row["temperature"] - 30) / 10 * 100, 0),
        100
    )

    # Green deficit score
    green_deficit = max(
        100 - row["green_cover"] * 3.33,
        0
    )

    # Pollution score
    pollution_scores = {
        "Low": 30,
        "Medium": 65,
        "High": 100
    }

    pollution_score = pollution_scores.get(
        row["pollution"],
        50
    )

    # Population exposure score
    population_scores = {
        "Low": 30,
        "Medium": 65,
        "High": 100
    }

    population_score = population_scores.get(
        row["population_density"],
        50
    )

    # Planting opportunity score
    planting_score = min(
        (row["plantable_area"] / 5000) * 100,
        100
    )

    # Final weighted score
    score = (
        heat_score * 0.25
        + green_deficit * 0.30
        + pollution_score * 0.15
        + population_score * 0.15
        + planting_score * 0.15
    )

    return round(score, 1)


def get_priority_label(score):
    """
    Convert numerical score into priority category.
    """

    if score >= 75:
        return "High"

    elif score >= 50:
        return "Medium"

    else:
        return "Low"


def analyze_zones(csv_path):
    """
    Load urban zone data and calculate
    priority score for every zone.
    """

    df = pd.read_csv(csv_path)

    df["priority_score"] = df.apply(
        calculate_priority_score,
        axis=1
    )

    df["priority"] = df["priority_score"].apply(
        get_priority_label
    )

    return df
def get_priority_breakdown(row):
    """
    Explain the individual factors contributing
    to the urban priority score.
    """

    heat_score = min(
        max((row["temperature"] - 30) / 10 * 100, 0),
        100
    )

    green_deficit = max(
        100 - row["green_cover"] * 3.33,
        0
    )

    pollution_scores = {
        "Low": 30,
        "Medium": 65,
        "High": 100
    }

    pollution_score = pollution_scores.get(
        row["pollution"],
        50
    )

    population_scores = {
        "Low": 30,
        "Medium": 65,
        "High": 100
    }

    population_score = population_scores.get(
        row["population_density"],
        50
    )

    planting_score = min(
        (row["plantable_area"] / 5000) * 100,
        100
    )

    return {
        "Heat Stress": round(heat_score, 1),
        "Green Deficit": round(green_deficit, 1),
        "Pollution": round(pollution_score, 1),
        "Population Exposure": round(population_score, 1),
        "Planting Opportunity": round(planting_score, 1)
    }