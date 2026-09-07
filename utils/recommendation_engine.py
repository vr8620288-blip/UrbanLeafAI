def recommend_trees(row):
    """
    AI-style multi-factor tree recommendation.

    The prototype scores different tree species according
    to urban conditions and selects the best match.
    """

    temperature = float(row["temperature"])
    green_cover = float(row["green_cover"])
    plantable_area = float(row["plantable_area"])

    pollution = str(row["pollution"])
    population = str(row["population_density"])


    # ========================================================
    # SPECIES PROFILES
    # ========================================================

    species_profiles = {

        "Neem": {
            "carbon_per_year": 0.060,
            "pollution_score": 1.0,
            "heat_score": 0.9,
            "urban_score": 1.0,
            "description":
                "Highly suitable for hot and polluted urban areas."
        },

        "Rain Tree": {
            "carbon_per_year": 0.070,
            "pollution_score": 0.7,
            "heat_score": 1.0,
            "urban_score": 0.8,
            "description":
                "Large canopy makes it useful for shade and heat reduction."
        },

        "Jamun": {
            "carbon_per_year": 0.055,
            "pollution_score": 0.8,
            "heat_score": 0.8,
            "urban_score": 0.9,
            "description":
                "Well suited to urban environments and community spaces."
        },

        "Indian Banyan": {
            "carbon_per_year": 0.080,
            "pollution_score": 0.6,
            "heat_score": 1.0,
            "urban_score": 0.5,
            "description":
                "Very high canopy and carbon potential but requires more space."
        }
    }


    # ========================================================
    # ENVIRONMENTAL FACTORS
    # ========================================================

    heat_need = min(
        max((temperature - 30) / 8, 0),
        1
    )

    vegetation_need = min(
        max((30 - green_cover) / 30, 0),
        1
    )

    pollution_need = {
        "Low": 0.3,
        "Medium": 0.65,
        "High": 1.0
    }.get(pollution, 0.5)


    # More available space means larger-canopy trees
    space_score = min(
        plantable_area / 5000,
        1
    )


    # ========================================================
    # SPECIES SCORING
    # ========================================================

    scores = {}

    for species, profile in species_profiles.items():

        score = (

            heat_need *
            profile["heat_score"] *
            0.30

            +

            vegetation_need *
            profile["urban_score"] *
            0.20

            +

            pollution_need *
            profile["pollution_score"] *
            0.25

            +

            space_score *
            profile["urban_score"] *
            0.15

            +

            profile["carbon_per_year"] /
            0.080 *
            0.10
        )

        scores[species] = score


    # ========================================================
    # BEST SPECIES
    # ========================================================

    recommended_species = max(
        scores,
        key=scores.get
    )

    profile = species_profiles[
        recommended_species
    ]


    # ========================================================
    # TREE COUNT
    # ========================================================

    estimated_trees = int(
        plantable_area / 10
    )

    estimated_trees = min(
        estimated_trees,
        1000
    )


    # ========================================================
    # CO2 ESTIMATION
    # ========================================================

    carbon_capture = (
        estimated_trees *
        profile["carbon_per_year"]
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "species":
            recommended_species,

        "trees":
            estimated_trees,

        "carbon_capture":
            round(
                carbon_capture,
                2
            ),

        "reason":
            profile["description"],

        "species_scores":
            {
                name:
                round(score * 100, 1)
                for name, score
                in scores.items()
            }
    }