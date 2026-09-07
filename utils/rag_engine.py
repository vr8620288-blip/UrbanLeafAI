from pathlib import Path
import re
import math


# ============================================================
# URBANLEAF AI — RAG ENGINE
# ============================================================

# Project root:
# UrbanLeafAI/
#     utils/rag_engine.py
#
# Therefore parent.parent = UrbanLeafAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text):
    """
    Convert text into simple normalized words.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
    )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_documents():
    """
    Load all .txt files from the UrbanLeaf knowledge base.
    """

    documents = []

    if not KNOWLEDGE_BASE_DIR.exists():
        return documents

    for file_path in sorted(
        KNOWLEDGE_BASE_DIR.glob("*.txt")
    ):

        try:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                    "tokens": tokenize(text),
                }
            )

        except Exception:
            continue

    return documents


# Load knowledge when the module starts
DOCUMENTS = load_documents()


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve(query, top_k=3):
    """
    Retrieve the most relevant knowledge-base documents.

    Uses:
    - keyword overlap
    - important query terms
    - document length normalization

    Returns a list of:
        source
        text
        score
    """

    if not DOCUMENTS:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return DOCUMENTS[:top_k]

    scored_documents = []

    # Common words that should not dominate retrieval
    stop_words = {
        "the",
        "is",
        "a",
        "an",
        "this",
        "that",
        "what",
        "why",
        "how",
        "can",
        "could",
        "should",
        "would",
        "here",
        "there",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
        "my",
        "we",
        "our",
        "it",
    }

    useful_query_tokens = (
        query_tokens - stop_words
    )

    if not useful_query_tokens:
        useful_query_tokens = query_tokens

    for document in DOCUMENTS:

        doc_tokens = document["tokens"]

        matches = (
            useful_query_tokens
            & doc_tokens
        )

        # Basic keyword relevance
        overlap_score = (
            len(matches)
            / max(len(useful_query_tokens), 1)
        )

        # Small bonus for exact phrase matches
        query_lower = query.lower()
        text_lower = document["text"].lower()

        phrase_bonus = 0.0

        if query_lower in text_lower:
            phrase_bonus = 0.5

        # Bonus for important UrbanLeaf terms
        important_terms = {
            "tree",
            "trees",
            "neem",
            "jamun",
            "rain",
            "banyan",
            "green",
            "greening",
            "pollution",
            "air",
            "quality",
            "pm25",
            "pm10",
            "temperature",
            "heat",
            "soil",
            "water",
            "municipality",
            "planting",
            "plantation",
            "carbon",
            "gis",
            "satellite",
            "iot",
            "priority",
            "zone",
        }

        important_matches = (
            useful_query_tokens
            & important_terms
            & doc_tokens
        )

        important_bonus = (
            len(important_matches) * 0.08
        )

        final_score = (
            overlap_score
            + phrase_bonus
            + important_bonus
        )

        scored_documents.append(
            {
                "source": document["source"],
                "text": document["text"],
                "score": round(
                    final_score,
                    3
                ),
            }
        )

    # Highest relevance first
    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # IMPORTANT:
    # Always return the best available documents.
    # This prevents the "No relevant knowledge"
    # problem when scores are low.
    return scored_documents[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    """
    Combine retrieved documents into LLM context.
    """

    if not results:
        return ""

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
SOURCE: {result['source']}

{result['text']}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# PROTOTYPE ANSWER
# ============================================================

def answer_from_context(
    question,
    context,
    zone=None,
    recommendation=None
):
    """
    Compatibility function from the earlier RAG prototype.

    The real Gemini LLM is now handled by llm_engine.py.
    """

    if not context:

        return (
            "No knowledge was retrieved from "
            "the UrbanLeaf knowledge base."
        )

    return (
        "UrbanLeaf AI found relevant information "
        "in its environmental knowledge base. "
        "The retrieved evidence can now be passed "
        "to the Gemini LLM for a grounded explanation."
    )


# ============================================================
# DEBUG / STATUS
# ============================================================

def get_knowledge_base_status():
    """
    Return useful diagnostic information.
    """

    return {
        "project_root": str(PROJECT_ROOT),
        "knowledge_base": str(
            KNOWLEDGE_BASE_DIR
        ),
        "exists": KNOWLEDGE_BASE_DIR.exists(),
        "documents": len(DOCUMENTS),
        "sources": [
            document["source"]
            for document in DOCUMENTS
        ],
    }