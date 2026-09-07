import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD .ENV FROM PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


def generate_grounded_answer(
    question,
    context,
    zone=None,
    recommendation=None
):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "⚠️ Gemini API key was not found.\n\n"
            f"Expected .env file at:\n{ENV_FILE}\n\n"
            "Make sure your .env contains:\n"
            "GEMINI_API_KEY=your_key"
        )

    try:

        client = genai.Client(
            api_key=api_key
        )

        # ----------------------------------------------------
        # ZONE DATA
        # ----------------------------------------------------

        zone_information = "No zone information available."

        if zone is not None:

            zone_information = f"""
Zone: {zone.get('zone', 'Unknown')}
Temperature: {zone.get('temperature', 'Unknown')} °C
Green Cover: {zone.get('green_cover', 'Unknown')}%
Pollution: {zone.get('pollution', 'Unknown')}
Plantable Area: {zone.get('plantable_area', 'Unknown')} m²
Population Density: {zone.get('population_density', 'Unknown')}
Priority Score: {zone.get('priority_score', 'Unknown')}
"""

        # ----------------------------------------------------
        # TREE RECOMMENDATION
        # ----------------------------------------------------

        recommendation_information = (
            "No tree recommendation available."
        )

        if recommendation is not None:

            recommendation_information = f"""
Species: {recommendation.get('species', 'Unknown')}
Trees: {recommendation.get('trees', 'Unknown')}
Reason: {recommendation.get('reason', 'Unknown')}
"""

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are UrbanLeaf AI.

UrbanLeaf AI is an urban-greening decision-support
platform that helps cities decide:

WHERE to intervene,
WHY an area is a priority,
WHAT trees may be suitable,
HOW MUCH planting may be appropriate,
and WHAT the possible impact could be.

USER QUESTION:
{question}

CURRENT ZONE DATA:
{zone_information}

TREE RECOMMENDATION:
{recommendation_information}

RETRIEVED KNOWLEDGE:
{context}

RULES:

1. Use the retrieved knowledge as supporting evidence.
2. Use the supplied zone data when relevant.
3. Never invent environmental measurements.
4. Do not present prototype estimates as guaranteed
   scientific predictions.
5. Clearly distinguish real data from estimates.
6. Explain WHY a recommendation was made.
7. If information is unavailable, say so.
8. Keep the answer concise and practical.
9. Do not discuss internal software implementation.

Give a natural, clear answer suitable for a
municipal officer or hackathon judge.
"""

        # ----------------------------------------------------
        # GEMINI REQUEST
        # ----------------------------------------------------

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response.text:

            return response.text.strip()

        return "UrbanLeaf AI could not generate an answer."

    except Exception as e:

        return (
            "⚠️ Gemini error:\n\n"
            f"{e}"
        )