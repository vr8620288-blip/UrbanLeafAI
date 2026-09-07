import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

def _get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
]

def _fallback_answer(question, zone, recommendation):
    name = str(zone.get("zone", "this zone"))
    score = float(zone.get("priority_score", 0))
    temp = float(zone.get("temperature", 0))
    green = float(zone.get("green_cover", 0))
    pollution = str(zone.get("pollution", "unknown")).lower()
    species = recommendation.get("species", "a suitable species")
    trees = int(recommendation.get("trees", 0))
    level = "high" if score >= 70 else "medium" if score >= 50 else "low"
    return (
        f"**{name} is currently a {level} priority zone** with an AI priority "
        f"score of {score:.0f}/100. The decision is supported by {temp:.1f}°C "
        f"temperature, {green:.1f}% green cover, and {pollution} pollution exposure. "
        f"UrbanLeaf recommends **{species}** with an estimated intervention of "
        f"about **{trees:,} trees**. Gemini is temporarily unavailable, so this "
        f"response is using UrbanLeaf's deterministic decision engine and knowledge base."
    )

def generate_grounded_answer(question, context, zone, recommendation):
    api_key = _get_api_key()
    if not api_key:
        return "⚠️ Gemini API key is not configured. Add GEMINI_API_KEY to Streamlit Cloud Secrets."

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are UrbanLeaf AI, an urban-greening decision-support copilot.
Answer using ONLY the supplied UrbanLeaf context and zone data.
Do not invent measurements. Be concise and practical.
Distinguish current data from future/prototype scenarios.

USER QUESTION:
{question}

SELECTED ZONE:
{zone.to_dict() if hasattr(zone, "to_dict") else zone}

TREE RECOMMENDATION:
{recommendation}

RETRIEVED URBANLEAF KNOWLEDGE:
{context}
"""

    for model in MODEL_CANDIDATES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                answer = getattr(response, "text", None)
                if answer:
                    return answer
            except Exception as exc:
                message = str(exc).lower()
                if any(x in message for x in ("503", "unavailable", "high demand", "429", "resource_exhausted", "quota")):
                    time.sleep(1.0 + attempt)
                    continue
                if any(x in message for x in ("404", "not found", "not supported")):
                    break
                if "401" in message or "403" in message or "api key" in message:
                    return "⚠️ Gemini authentication/configuration error. Check GEMINI_API_KEY in Streamlit Cloud Secrets."
                break

    return _fallback_answer(question, zone, recommendation)

