import io
import speech_recognition as sr
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

from utils.priority_engine import analyze_zones, get_priority_breakdown
from utils.recommendation_engine import recommend_trees
from utils.rag_engine import retrieve, answer_from_context
from utils.llm_engine import generate_grounded_answer
from utils.location_engine import (
    find_nearest_zone,
    get_zone_name,
    get_priority_level,
    get_priority_score,
)
# Real-world air-quality data layer
try:
    from utils.zone_air_quality import match_zones_to_air_quality
except Exception:
    match_zones_to_air_quality = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="UrbanLeaf AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — COMPACT / ONE-SCREEN DASHBOARD
# ============================================================
st.markdown(
    """
<style>
.stApp { background:#f5faf7; color:#183c2c; }
.block-container { padding-top:0.75rem; padding-bottom:1.0rem; max-width:1500px; }
h1,h2,h3,h4 { color:#075e3c !important; }
p,label { color:#365b4a; }

/* Compact hero */
.hero {
    background:linear-gradient(135deg,#e6f7ec,#d5f0df);
    border:1px solid #c3e5cf; border-radius:18px;
    padding:15px 20px; margin-bottom:10px;
    box-shadow:0 5px 18px rgba(20,100,60,.06);
}
.hero-title { font-size:31px; font-weight:850; color:#075e3c; line-height:1.05; }
.hero-sub { font-size:13px; color:#4c7562; margin-top:4px; }
.badge { display:inline-block; margin-top:7px; padding:4px 9px; border-radius:99px;
    background:white; border:1px solid #c7e6d2; color:#087f4f; font-size:11px; font-weight:750; }

/* KPI */
[data-testid="stMetric"] { background:#fff; border:1px solid #d9ebe1; border-radius:12px;
    padding:9px 12px; box-shadow:0 3px 12px rgba(20,100,60,.045); }
[data-testid="stMetricLabel"] { color:#648072 !important; font-size:12px !important; }
[data-testid="stMetricValue"] { color:#087f4f !important; font-size:24px !important; font-weight:800; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#e4f6eb; border-right:1px solid #c9e6d3; }
section[data-testid="stSidebar"] * { color:#164b34 !important; }
section[data-testid="stSidebar"] h1 { color:#075e3c !important; }

/* Nav radio */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding:7px 9px; border-radius:9px; margin:2px 0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#d3efdd; }

/* Compact cards */
.card { background:#fff; border:1px solid #d9ebe1; border-radius:14px; padding:13px;
    box-shadow:0 3px 12px rgba(20,100,60,.045); height:100%; }
.card-title { color:#075e3c; font-weight:800; font-size:15px; margin-bottom:6px; }
.mini { font-size:12px; color:#5a7768; }
.priority-high { color:#c94141; font-weight:850; }
.priority-medium { color:#b47a00; font-weight:850; }
.priority-low { color:#16834e; font-weight:850; }
.big-score { font-size:36px; font-weight:850; color:#087f4f; line-height:1; }
.action { background:#edf9f1; border:1px solid #cbe8d4; border-radius:11px; padding:9px 11px; margin-top:7px; }
.report-box { background:#f8fcf9; border:1px dashed #b8dcca; border-radius:13px; padding:13px; }

/* Remove excess vertical spacing */
.stMarkdown { margin-bottom:0.2rem; }
hr { border-color:#d8eae0; margin:8px 0; }
[data-testid="stDataFrame"] { border:1px solid #d9ebe1; border-radius:10px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
DATA_PATH = "data/urban_zones.csv"
df = analyze_zones(DATA_PATH).sort_values("priority_score", ascending=False).reset_index(drop=True)

# Load cached real OpenAQ zone matches.
# The app remains usable if the real-data cache is unavailable.
REAL_AQ_PATH = "data/real_data/zone_air_quality.csv"
try:
    real_aq_df = pd.read_csv(REAL_AQ_PATH) if Path(REAL_AQ_PATH).exists() else pd.DataFrame()
except Exception:
    real_aq_df = pd.DataFrame()

if not real_aq_df.empty and "zone" in real_aq_df.columns:
    df = df.merge(real_aq_df, on="zone", how="left", suffixes=("", "_real"))


def priority_class(score):
    if score >= 75:
        return "HIGH", "priority-high"
    if score >= 50:
        return "MEDIUM", "priority-medium"
    return "LOW", "priority-low"


def map_color(priority):
    return {"High": [220, 70, 70, 200], "Medium": [240, 180, 50, 200], "Low": [60, 180, 110, 200]}.get(priority, [60,180,110,200])


def build_map(data, height=330, selected_zone_name=None):
    """Interactive city map: clicking a marker changes the active zone."""
    map_df = data.copy()

    map_df["is_selected"] = (
        map_df["zone"].astype(str) == str(selected_zone_name)
        if selected_zone_name is not None else False
    )
    map_df["real_aq_status"] = map_df.apply(real_aq_status, axis=1)
    map_df["real_aq_flag"] = map_df["real_aq_status"].apply(
        lambda x: "🌍 REAL AQ DATA"
        if str(x).startswith("REAL AQ") else "AQ cache unavailable"
    )

    map_df["color"] = map_df["priority"].apply(map_color)
    selected_count = int(map_df["is_selected"].sum())
    if selected_count:
        # Pandas treats a list like [r,g,b,a] as a 2-D ndarray when
        # assigning to a single column. Use one list object per selected row.
        map_df.loc[map_df["is_selected"], "color"] = map_df.loc[
            map_df["is_selected"], "color"
        ].apply(lambda _: [0, 126, 86, 235])

    map_df["radius"] = map_df["priority_score"] * 11
    if selected_count:
        map_df.loc[map_df["is_selected"], "radius"] = (
            map_df.loc[map_df["is_selected"], "priority_score"] * 20
        ).clip(lower=300)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        id="city-priority-zones",
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=42,
        pickable=True,
        auto_highlight=True,
    )

    selected_rows = map_df[map_df["is_selected"]]
    if not selected_rows.empty:
        center_lat = float(selected_rows.iloc[0]["latitude"])
        center_lon = float(selected_rows.iloc[0]["longitude"])
        zoom = 13
    else:
        center_lat, center_lon, zoom = 18.54, 73.87, 10.35

    view = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{zone}</b><br/>"
            "AI Priority: {priority_score}<br/>"
            "Temperature: {temperature} °C<br/>"
            "Green Cover: {green_cover}%<br/>"
            "Pollution: {pollution}<br/>"
            "Priority: {priority}<br/><br/>"
            "<b>{real_aq_flag}</b><br/>{real_aq_status}"
        ),
        "style": {
            "backgroundColor": "#ffffff",
            "color": "#183c2c",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        map_style=None,
        initial_view_state=view,
        layers=[layer],
        tooltip=tooltip,
    ), height


def species_library():
    return {
        "Neem": {"best_for":"Pollution exposure • heat-tolerant urban spaces","strength":"High urban resilience","carbon":0.060,"space":"Medium","reason":"Strong default for hot, polluted urban environments."},
        "Rain Tree": {"best_for":"High heat • larger open spaces","strength":"Large canopy potential","carbon":0.080,"space":"High","reason":"Useful where there is enough space for a broad canopy."},
        "Jamun": {"best_for":"Mixed urban conditions • public spaces","strength":"Urban adaptability","carbon":0.065,"space":"Medium","reason":"A versatile option for mixed environmental conditions."},
        "Indian Banyan": {"best_for":"Very large open spaces • long-term canopy","strength":"Very large canopy potential","carbon":0.090,"space":"Very High","reason":"Best considered only where substantial space is available."},
    }


def species_scores(zone):
    profiles = species_library()
    temp, green, plantable = float(zone["temperature"]), float(zone["green_cover"]), float(zone["plantable_area"])
    pollution = str(zone["pollution"]).lower()
    scores = {}
    for species in profiles:
        score = 55.0
        if temp >= 36: score += 14 if species in ["Neem","Rain Tree"] else 7
        elif temp >= 33: score += 9 if species in ["Neem","Jamun"] else 6
        if green <= 20: score += 10
        elif green <= 30: score += 6
        if pollution == "high": score += 10 if species == "Neem" else 5
        elif pollution == "medium": score += 6 if species in ["Neem","Jamun"] else 3
        if plantable >= 2500 and species in ["Rain Tree","Indian Banyan"]: score += 8
        elif plantable < 1500 and species == "Indian Banyan": score -= 18
        scores[species] = max(0, min(100, score))
    return scores


def focus_areas(zone):
    breakdown = get_priority_breakdown(zone)
    labels = {"Temperature":"heat stress","Green Cover":"vegetation deficit","Pollution":"pollution exposure","Population":"population exposure","Plantable Area":"available planting space"}
    ranked = sorted(breakdown.items(), key=lambda x:x[1], reverse=True)
    return [labels.get(k,k.lower()) for k,_ in ranked[:3]]


def get_real_aq(zone):
    """Return real OpenAQ evidence for the selected zone."""
    if real_aq_df.empty or "zone" not in real_aq_df.columns:
        return {}

    rows = real_aq_df[real_aq_df["zone"].astype(str) == str(zone["zone"])]
    if rows.empty:
        return {}

    row = rows.iloc[0].to_dict()

    # Only expose pollutant fields that actually exist and have values.
    clean = {}
    for key in ["nearest_station", "distance_km", "pm25", "pm25_units",
                "pm10", "pm10_units", "no2", "no2_units",
                "so2", "so2_units", "co", "co_units",
                "o3", "o3_units"]:
        value = row.get(key)
        if pd.notna(value):
            clean[key] = value

    return clean


def render_real_air_quality(zone, compact=False):
    """Render cached real-world OpenAQ evidence."""
    aq = get_real_aq(zone)

    if not aq:
        st.info("🌍 Real air-quality cache is not available for this zone yet.")
        return

    st.markdown("### 🌍 Real Air Quality Evidence")
    station = aq.get("nearest_station", "OpenAQ monitoring station")
    distance = aq.get("distance_km")

    if distance is not None:
        st.caption(
            f"Source: OpenAQ • Nearest station: **{station}** • "
            f"Distance: **{float(distance):.2f} km**"
        )
    else:
        st.caption(f"Source: OpenAQ • Nearest station: **{station}**")

    pollutant_items = []
    labels = [
        ("pm25", "PM2.5"),
        ("pm10", "PM10"),
        ("no2", "NO₂"),
        ("so2", "SO₂"),
        ("co", "CO"),
        ("o3", "O₃"),
    ]

    for key, label in labels:
        if key in aq:
            unit = aq.get(f"{key}_units", "")
            unit_text = f" {unit}" if unit else ""
            pollutant_items.append((label, aq[key], unit_text))

    if pollutant_items:
        cols = st.columns(min(4, len(pollutant_items)))
        for i, (label, value, unit_text) in enumerate(pollutant_items):
            with cols[i % len(cols)]:
                st.metric(label, f"{float(value):.2f}{unit_text}")
    else:
        st.warning("The station was matched, but no supported pollutant field was found.")

    st.caption(
        "Real monitoring evidence is shown separately from UrbanLeaf's "
        "prototype priority/scenario calculations."
    )


def make_visual_pdf(zone, recommendation, breakdown, focus):
    """Create a compact visual municipal action-card PDF with diagrams."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=20, leading=23, textColor=colors.HexColor("#075e3c"), spaceAfter=3)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#527466"), leading=12)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12, leading=14, textColor=colors.HexColor("#075e3c"), spaceBefore=5, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#284c3c"))
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9)

    score = float(zone["priority_score"])
    level = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
    level_color = {"HIGH":"#c94141","MEDIUM":"#b47a00","LOW":"#16834e"}[level]

    story = [Paragraph("URBANLEAF AI", title), Paragraph("Municipal Urban Greening Action Card • Visual Decision-Support Report", sub), Spacer(1, 5)]

    # Header diagram
    header_data = [[
        Paragraph(f"<b>ZONE</b><br/><font size='15'>{zone['zone']}</font>", body),
        Paragraph(f"<b>AI PRIORITY</b><br/><font size='17' color='{level_color}'>{level}</font><br/>{score:.0f}/100", body),
        Paragraph(f"<b>AI ACTION</b><br/><font size='15'>{recommendation['trees']:,} trees</font><br/>{recommendation['species']}", body),
    ]]
    header = Table(header_data, colWidths=[58*mm, 48*mm, 68*mm], rowHeights=[28*mm])
    header.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#edf8f1")),
        ("BOX",(0,0),(-1,-1),0.7,colors.HexColor("#b9dec7")),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.HexColor("#cfe7d7")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story += [header, Spacer(1, 7)]

    story.append(Paragraph("1. ENVIRONMENTAL SNAPSHOT", h))
    snap = [["Temperature", f"{float(zone['temperature']):.1f} °C", "Green Cover", f"{float(zone['green_cover']):.1f}%"],
            ["Pollution", str(zone["pollution"]), "Plantable Area", f"{float(zone['plantable_area']):,.0f} m²"],
            ["Population", str(zone["population_density"]), "Existing Trees", f"{int(zone['existing_trees']):,}"]]
    t = Table(snap, colWidths=[34*mm,52*mm,34*mm,54*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.white), ("GRID",(0,0),(-1,-1),0.45,colors.HexColor("#d4e8dc")),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#527466")), ("TEXTCOLOR",(2,0),(2,-1),colors.HexColor("#527466")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("PADDING",(0,0),(-1,-1),6),
    ]))
    story += [t, Spacer(1,6)]

    story.append(Paragraph("2. WHY THIS ZONE?", h))
    bars=[]
    for name,val in sorted(breakdown.items(), key=lambda x:x[1], reverse=True):
        bars.append([Paragraph(f"<b>{name}</b>", small), Paragraph(f"{val:.0f}/100", small), ""])
    bt=Table(bars, colWidths=[48*mm,22*mm,70*mm], rowHeights=[7*mm]*len(bars))
    bt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f7fbf8")), ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#dbece1")),
        ("ALIGN",(1,0),(1,-1),"RIGHT"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("PADDING",(0,0),(-1,-1),4),
    ]))
    story += [bt, Spacer(1,6)]

    story.append(Paragraph("3. RECOMMENDED INTERVENTION", h))
    actions = [["🌳 Plant", f"Approximately {recommendation['trees']:,} trees"],
               ["🌱 Species", f"{recommendation['species']} — preferred AI match"],
               ["📐 Area", f"Approximately {float(zone['plantable_area']):,.0f} m²"],
               ["🌍 CO₂", f"Prototype scenario: {float(recommendation['carbon_capture']):.1f} t/year"],
               ["🎯 Focus", ", ".join(focus).title()]]
    at=Table(actions, colWidths=[32*mm,140*mm])
    at.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#edf8f1")), ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cfe7d7")),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.8), ("PADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story += [at, Spacer(1,6)]

    story.append(Paragraph("4. DECISION FLOW", h))
    flow=Table([["DATA", "→", "AI PRIORITY", "→", "TREE MATCH", "→", "ACTION", "→", "IMPACT"]], colWidths=[25*mm,7*mm,29*mm,7*mm,29*mm,7*mm,25*mm,7*mm,28*mm])
    flow.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#e8f6ed")), ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#b9dec7")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#cfe7d7")), ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("FONTSIZE",(0,0),(-1,-1),7.5), ("PADDING",(0,0),(-1,-1),5),
    ]))
    story += [flow, Spacer(1,6), Paragraph("Decision-support note: environmental and CO₂ figures are prototype scenario estimates, not certified measurements. Production deployment should calibrate coefficients using local GIS, climate, ecology, tree-growth and environmental datasets.", small)]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def real_aq_status(zone):
    """Return a compact real-data status for map/tooltips."""
    aq = get_real_aq(zone)
    if not aq:
        return "No cached AQ match"

    station = aq.get("nearest_station", "OpenAQ station")
    distance = aq.get("distance_km")
    if distance is not None:
        return f"REAL AQ • {station} • {float(distance):.1f} km"
    return f"REAL AQ • {station}"



# ============================================================
# SIDEBAR NAV
# ============================================================
# ============================================================
# SIDEBAR NAV
# ============================================================

with st.sidebar:

    st.markdown("# 🌱 UrbanLeaf AI")

    st.caption(
        "Greener Cities. Healthier Tomorrows."
    )

    st.divider()

    nav = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📍 Analyze My Location",
            "🗺️ City Map",
            "📊 Zone Analysis",
            "🤖 AI Copilot",
            "🔮 What-If Simulator",
            "🌳 Tree Library",
            "📄 Reports",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.caption(
        "AI-powered urban greening decision support"
    )

    st.caption(
        "Version 1.0 • Offline MVP"
    )
# ============================================================
# COMMON SELECTION
# ============================================================
zone_names = df["zone"].tolist()
if "selected_zone_name" not in st.session_state:
    st.session_state.selected_zone_name = zone_names[0]

# Small selector in sidebar is always available.
with st.sidebar:
    st.session_state.selected_zone_name = st.selectbox(
        "ACTIVE ZONE", zone_names,
        index=zone_names.index(st.session_state.selected_zone_name),
    )

selected_zone = df[df["zone"] == st.session_state.selected_zone_name].iloc[0]
recommendation = recommend_trees(selected_zone)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""<div class='hero'><div class='hero-title'>🌱 UrbanLeaf AI</div>
    <div class='hero-sub'>AI-powered urban greening & carbon action platform • <b>{selected_zone['zone']}</b> active</div>
    <span class='badge'>● OFFLINE HACKATHON MVP • DECISION SUPPORT</span></div>""",
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD — DESIGNED TO FIT ONE SCREEN
# ============================================================
if nav == "🏠 Dashboard":
    total_zones = len(df)
    high_priority = int((df["priority"] == "High").sum())
    total_plantable = float(df["plantable_area"].sum())
    estimated_carbon = float(df["existing_trees"].sum()) * 0.060

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🌳 Zones Analyzed", total_zones)
    with c2: st.metric("🔴 High Priority", high_priority)
    with c3: st.metric("🌿 Plantable Area", f"{total_plantable/10000:.1f} ha")
    with c4: st.metric("🌍 Existing CO₂", f"{estimated_carbon:.0f} t/yr")

    st.write("")
    left,right = st.columns([1.7,1])

    with left:
        st.markdown(
            f"<div class='card'><div class='card-title'>🗺️ City Priority Map • Active: {selected_zone['zone']}</div>",
            unsafe_allow_html=True,
        )

        deck, height = build_map(
            df, 340, selected_zone["zone"]
        )

        map_event = st.pydeck_chart(
            deck,
            height=height,
            selection_mode="single-object",
            on_select="rerun",
            key="dashboard_city_priority_map",
        )

        try:
            selected_objects = map_event.selection.objects.get(
                "city-priority-zones", []
            )
        except Exception:
            selected_objects = []

        if selected_objects:
            clicked_zone = selected_objects[0].get("zone")
            if (
                clicked_zone in zone_names
                and clicked_zone != st.session_state.selected_zone_name
            ):
                st.session_state.selected_zone_name = clicked_zone
                st.rerun()

        st.caption(
            "🔴 High   🟡 Medium   🟢 Low • "
            "Click any zone marker to update the active location."
        )
        st.caption(
            "🌍 Each selected zone uses its matched OpenAQ evidence "
            "where available."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        level, cls = priority_class(float(selected_zone["priority_score"]))
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>🎯 {selected_zone['zone']} — AI Decision</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-score'>{float(selected_zone['priority_score']):.0f}<span style='font-size:15px'> / 100</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='{cls}'>{level} PRIORITY</div>", unsafe_allow_html=True)
        st.write("")
        k1,k2 = st.columns(2)
        with k1: st.metric("🌡️ Temp", f"{selected_zone['temperature']:.1f} °C")
        with k2: st.metric("🌿 Green", f"{selected_zone['green_cover']:.1f}%")
        k3,k4 = st.columns(2)
        with k3: st.metric("🌳 AI Trees", f"{recommendation['trees']:,}")
        with k4: st.metric("🌍 CO₂", f"{recommendation['carbon_capture']:.1f} t/yr")
        st.markdown(f"<div class='action'><b>🌱 Recommended:</b> {recommendation['species']}<br/><span class='mini'>{recommendation['reason']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='action'><b>⚠️ Focus:</b> {', '.join(focus_areas(selected_zone)).title()}</div>", unsafe_allow_html=True)

        aq = get_real_aq(selected_zone)
        if aq:
            station = aq.get("nearest_station", "OpenAQ")
            pm25 = aq.get("pm25")
            pm10 = aq.get("pm10")
            aq_line = f"PM2.5: {float(pm25):.1f}" if pm25 is not None else ""
            if pm10 is not None:
                aq_line += f" • PM10: {float(pm10):.1f}"
            st.markdown(
                f"<div class='action'><b>🌍 Real AQ:</b> {aq_line}<br/>"
                f"<span class='mini'>OpenAQ • {station}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f"### 🌍 Live Real Evidence — {selected_zone['zone']}"
    )

    selected_aq = get_real_aq(selected_zone)

    if selected_aq:
        station = selected_aq.get("nearest_station", "OpenAQ")
        distance = selected_aq.get("distance_km")

        evidence = []
        for key, label in [
            ("pm25", "PM2.5"),
            ("pm10", "PM10"),
            ("no2", "NO₂"),
            ("so2", "SO₂"),
        ]:
            if key in selected_aq:
                try:
                    value = f"{float(selected_aq[key]):.2f}"
                except Exception:
                    value = str(selected_aq[key])
                evidence.append((label, value))

        if evidence:
            eco = st.columns(min(4, len(evidence)))
            for i, (label, value) in enumerate(evidence):
                with eco[i]:
                    st.metric(f"🌍 {label}", value)

        distance_text = (
            f"{float(distance):.2f} km away"
            if distance is not None
            else "nearest matched station"
        )

        st.caption(
            f"OpenAQ • {station} • {distance_text} • "
            "Nearby monitoring-station evidence, not a phone measurement."
        )
    else:
        st.info(
            f"No cached OpenAQ match is available for {selected_zone['zone']}."
        )

    st.write("")
    b1,b2,b3 = st.columns(3)
    top = df.head(3)
    for col,(_,row) in zip([b1,b2,b3], top.iterrows()):
        lvl,_ = priority_class(float(row["priority_score"]))
        with col:
            st.markdown(f"<div class='card'><div class='card-title'>#{int(row.name)+1} {row['zone']}</div><b>{lvl}</b> • {float(row['priority_score']):.0f}/100<br/><span class='mini'>Green cover {row['green_cover']:.1f}% • {row['pollution']} pollution</span></div>", unsafe_allow_html=True)

# ============================================================
# ANALYZE MY LOCATION
# ============================================================

elif nav == "📍 Analyze My Location":

    from streamlit_geolocation import streamlit_geolocation

    st.subheader("📍 Analyze My Location")

    st.caption(
        "Use your phone's GPS to identify the nearest "
        "UrbanLeaf zone and assess its greening priority."
    )

    st.info(
        "📱 Stand at a real location such as a roadside, "
        "parking area, campus or open urban space, then "
        "allow location access."
    )

    # --------------------------------------------------------
    # GET GPS LOCATION
    # --------------------------------------------------------

    location = streamlit_geolocation()

    if location is None:
        st.warning("Waiting for location access...")
        st.stop()

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    accuracy = location.get("accuracy")

    if latitude is None or longitude is None:
        st.warning(
            "📍 Location not available yet. "
            "Allow browser location access and try again."
        )
        st.stop()

    # --------------------------------------------------------
    # LOCATION DETAILS
    # --------------------------------------------------------

    st.success("📍 Your location has been detected.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Latitude", f"{latitude:.6f}")

    with c2:
        st.metric("Longitude", f"{longitude:.6f}")

    with c3:
        if accuracy is not None:
            st.metric("GPS Accuracy", f"{accuracy:.0f} m")
        else:
            st.metric("GPS Accuracy", "Available")

    # --------------------------------------------------------
    # FIND NEAREST URBANLEAF ZONE
    # --------------------------------------------------------

    try:
        nearest_zone, distance_km = find_nearest_zone(
            latitude,
            longitude,
            df,
        )
    except Exception as e:
        st.error(f"Unable to analyze this location: {e}")
        st.stop()

    zone_name = get_zone_name(nearest_zone)
    priority_level = get_priority_level(nearest_zone)
    priority_score = get_priority_score(nearest_zone)

    # --------------------------------------------------------
    # LOCATION ANALYSIS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🌱 Location Analysis")

    st.write(
        f"Your location is closest to **{zone_name}**."
    )

    st.caption(
        f"Nearest zone distance: {distance_km:.2f} km"
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if priority_score is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.metric("🌱 Greening Priority", priority_level)

        with c2:
            st.metric(
                "AI Priority Score",
                f"{priority_score:.1f}",
            )
    else:
        st.metric("🌱 Greening Priority", priority_level)

    # --------------------------------------------------------
    # WHY THIS LOCATION?
    # --------------------------------------------------------

    st.subheader("🔎 Why is this location important?")

    reasons = []

    if "temperature" in nearest_zone.index:
        try:
            reasons.append(
                f"🌡️ Temperature / heat indicator: "
                f"{float(nearest_zone['temperature']):.1f} °C"
            )
        except Exception:
            pass

    if "green_cover" in nearest_zone.index:
        try:
            reasons.append(
                f"🌳 Existing green cover: "
                f"{float(nearest_zone['green_cover']):.1f}%"
            )
        except Exception:
            pass

    if "pollution" in nearest_zone.index:
        reasons.append(
            f"💨 Pollution level: "
            f"{nearest_zone['pollution']}"
        )

    if "plantable_area" in nearest_zone.index:
        try:
            reasons.append(
                f"🌱 Plantable area: "
                f"{float(nearest_zone['plantable_area']):,.0f} m²"
            )
        except Exception:
            pass

    if reasons:
        for reason in reasons:
            st.write(reason)
    else:
        st.write(
            "🌱 This location is being evaluated using "
            "UrbanLeaf's available zone-level data."
        )

    # --------------------------------------------------------
    # TREE RECOMMENDATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🌳 Recommended Plantation")

    try:
        location_recommendation = recommend_trees(nearest_zone)

        st.success(
            f"🌳 **{location_recommendation['species']}**"
        )

        st.write(location_recommendation["reason"])

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Estimated Trees",
                f"{location_recommendation['trees']:,}",
            )

        with c2:
            st.metric(
                "CO₂ Scenario",
                f"{location_recommendation['carbon_capture']:.1f} t/year",
            )

    except Exception:
        st.info(
            "🌳 Tree recommendation uses heat, "
            "green-cover deficit, pollution, "
            "planting space and carbon potential."
        )

    # --------------------------------------------------------
    # REAL AIR QUALITY
    # --------------------------------------------------------

    location_aq = get_real_aq(nearest_zone)

    if location_aq:
        st.divider()

        st.subheader("🌍 Nearby Real Air-Quality Evidence")

        station = location_aq.get(
            "nearest_station",
            "OpenAQ monitoring station",
        )

        distance = location_aq.get("distance_km")

        if distance is not None:
            st.caption(
                f"Source: OpenAQ • Nearest station: "
                f"**{station}** • Distance: "
                f"**{float(distance):.2f} km**"
            )
        else:
            st.caption(
                f"Source: OpenAQ • Nearest station: "
                f"**{station}**"
            )

        pollutants = []

        for key, label in [
            ("pm25", "PM2.5"),
            ("pm10", "PM10"),
            ("no2", "NO₂"),
            ("so2", "SO₂"),
            ("co", "CO"),
            ("o3", "O₃"),
        ]:
            if key in location_aq:
                pollutants.append(
                    (label, location_aq[key])
                )

        if pollutants:
            cols = st.columns(min(4, len(pollutants)))

            for i, (label, value) in enumerate(pollutants):
                with cols[i % len(cols)]:
                    try:
                        display_value = f"{float(value):.2f}"
                    except Exception:
                        display_value = str(value)

                    st.metric(label, display_value)

        st.caption(
            "These are nearby monitoring-station observations, "
            "not direct measurements from your phone."
        )

    # --------------------------------------------------------
    # LIVE LOCATION + URBANLEAF ZONE MAP
    # --------------------------------------------------------

    st.divider()

    st.subheader("🗺️ Live Location + UrbanLeaf Zone")

    st.caption(
        "📍 Your GPS location • 🌱 Nearest UrbanLeaf analysis zone"
    )

    # Get nearest zone coordinates
    zone_lat = float(nearest_zone["latitude"])
    zone_lon = float(nearest_zone["longitude"])

    map_data = pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "type": "📍 Your Location",
                "name": "Your Current Location",
            },
            {
                "latitude": zone_lat,
                "longitude": zone_lon,
                "type": "🌱 UrbanLeaf Zone",
                "name": zone_name,
            },
        ]
    )

    map_data["color"] = map_data["type"].apply(
        lambda x: [220, 50, 50, 230]
        if "Your" in x
        else [40, 170, 90, 230]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=90,
        radius_min_pixels=8,
        radius_max_pixels=25,
        pickable=True,
        auto_highlight=True,
    )

    view = pdk.ViewState(
        latitude=float(latitude),
        longitude=float(longitude),
        zoom=13,
        pitch=0,
    )

    tooltip = {
        "html": """
            <b>{name}</b><br/>
            {type}
        """,
        "style": {
            "backgroundColor": "#ffffff",
            "color": "#183c2c",
            "fontSize": "13px",
        },
    }

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view,
        layers=[layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, height=450)

    st.info(
        f"📍 You are approximately **{distance_km:.2f} km** "
        f"from the **{zone_name}** UrbanLeaf analysis zone."
    )

    # --------------------------------------------------------
    # PROTOTYPE NOTE
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "⚡ Prototype: GPS identifies your location and maps "
        "it to the nearest UrbanLeaf zone. Production can "
        "improve precision using GIS, satellite imagery, "
        "soil data and IoT environmental sensors."
    )


# ============================================================
# CITY MAP
# ============================================================
elif nav == "🗺️ City Map":
    st.subheader("🗺️ Pune Urban Greening Priority Map")

    deck, _ = build_map(
        df, 560, selected_zone["zone"]
    )

    city_map_event = st.pydeck_chart(
        deck,
        height=560,
        selection_mode="single-object",
        on_select="rerun",
        key="city_map_zone_selector",
    )

    try:
        city_selected_objects = city_map_event.selection.objects.get(
            "city-priority-zones", []
        )
    except Exception:
        city_selected_objects = []

    if city_selected_objects:
        clicked_zone = city_selected_objects[0].get("zone")
        if (
            clicked_zone in zone_names
            and clicked_zone != st.session_state.selected_zone_name
        ):
            st.session_state.selected_zone_name = clicked_zone
            st.rerun()

    st.info(
        f"🎯 Active zone: **{selected_zone['zone']}** • "
        "Click any marker to update the UrbanLeaf decision."
    )

    map_table_cols = ["zone","priority_score","priority","temperature","green_cover","pollution"]
    if "nearest_station" in df.columns:
        map_table_cols += ["nearest_station","distance_km"]
    st.dataframe(
        df[map_table_cols].rename(
            columns={
                "zone":"Zone",
                "priority_score":"AI Score",
                "priority":"Priority",
                "temperature":"Temp °C",
                "green_cover":"Green Cover %",
                "pollution":"Pollution",
                "nearest_station":"Real AQ Station",
                "distance_km":"AQ Distance (km)"
            }
        ),
        width="stretch",
        hide_index=True
    )

# ============================================================
# ZONE ANALYSIS
# ============================================================
elif nav == "📊 Zone Analysis":
    st.subheader(f"📊 Zone Analysis — {selected_zone['zone']}")
    score=float(selected_zone["priority_score"]); level,cls=priority_class(score)
    a,b,c,d=st.columns(4)
    with a: st.metric("AI Priority", f"{score:.0f}/100")
    with b: st.metric("Temperature", f"{selected_zone['temperature']:.1f} °C")
    with c: st.metric("Plantable Area", f"{selected_zone['plantable_area']:,.0f} m²")
    with d: st.metric("AI Trees", f"{recommendation['trees']:,}")
    left,right=st.columns(2)
    with left:
        st.markdown("### 🧠 Why this zone?")
        for factor,value in get_priority_breakdown(selected_zone).items():
            st.write(f"**{factor} — {value:.0f}/100**")
            st.progress(int(value))
    with right:
        st.markdown("### 🌳 AI Plantation Recommendation")
        st.success(f"**{recommendation['species']}** — preferred match")
        st.write(recommendation["reason"])
        st.metric("Estimated CO₂ scenario", f"{recommendation['carbon_capture']:.1f} t/year")
        st.info(f"Primary focus: {', '.join(focus_areas(selected_zone)).title()}")
    render_real_air_quality(selected_zone)

    st.markdown("### 🌱 Action Plan")
    st.write(f"1. Prioritize **{selected_zone['zone']}** for intervention.")
    st.write(f"2. Target approximately **{selected_zone['plantable_area']:,.0f} m²** of planting space.")
    st.write(f"3. Introduce approximately **{recommendation['trees']:,} trees**, led by **{recommendation['species']}**.")
    st.write(f"4. Track the intervention against the prototype **{recommendation['carbon_capture']:.1f} t/year CO₂** scenario.")

# ============================================================
# AI COPILOT — RAG FOUNDATION
# ============================================================
# ============================================================
# AI COPILOT — RAG + GEMINI LLM
# ============================================================
# ============================================================
# AI COPILOT — TEXT + VOICE
# ============================================================
elif nav == "🤖 AI Copilot":

    st.subheader("🤖 UrbanLeaf AI Copilot")

    st.caption(
        f"Ask UrbanLeaf about **{selected_zone['zone']}** "
        "using environmental data + RAG + Gemini AI."
    )

    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    st.markdown("### 💬 Ask by Text")

    examples = [
        "Why is this zone a high priority?",
        f"Why is {recommendation['species']} recommended here?",
        "What data supports this decision?",
        "What should the municipality do first?",
        "How can GIS and satellite data improve this?",
        "How can soil sensors improve tree selection?",
    ]

    example_question = st.selectbox(
        "💡 Try an example",
        examples
    )

    question = st.text_input(
        "Ask UrbanLeaf AI",
        value=example_question,
        placeholder="Ask something about this zone..."
    )

    # --------------------------------------------------------
    # VOICE INPUT
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### 🎤 Ask by Voice")

    st.caption(
        "Tap the microphone and ask your question."
    )

    audio = st.audio_input(
        "🎤 Record your question"
    )

    voice_question = None

    if audio is not None:

        try:

            recognizer = sr.Recognizer()

            audio_bytes = audio.getvalue()

            with sr.AudioFile(
                io.BytesIO(audio_bytes)
            ) as source:

                recorded_audio = recognizer.record(source)

            with st.spinner(
                "🎧 Converting your voice to text..."
            ):

                voice_question = recognizer.recognize_google(
                    recorded_audio
                )

            st.success(
                f"🗣️ You said: **{voice_question}**"
            )

        except sr.UnknownValueError:

            st.error(
                "I couldn't understand the audio. "
                "Please speak clearly and try again."
            )

        except sr.RequestError as e:

            st.error(
                f"Speech recognition service error: {e}"
            )

        except Exception as e:

            st.error(
                f"Voice processing error: {e}"
            )

    # --------------------------------------------------------
    # SELECT QUESTION
    # --------------------------------------------------------

    final_question = voice_question or question

    # --------------------------------------------------------
    # ASK AI BUTTON
    # --------------------------------------------------------

    if st.button(
        "🤖 Ask UrbanLeaf AI",
        type="primary",
        width="stretch"
    ):

        if not final_question.strip():

            st.warning(
                "Please enter a question or record your voice."
            )

        else:

            with st.spinner(
                "🌱 UrbanLeaf is analyzing your question..."
            ):

                # --------------------------------------------
                # RAG RETRIEVAL
                # --------------------------------------------

                results = retrieve(
                    final_question,
                    top_k=3
                )

                context = "\n\n".join(
                    item["text"]
                    for item in results
                )

                # --------------------------------------------
                # GEMINI
                # --------------------------------------------

                answer = generate_grounded_answer(
                    question=final_question,
                    context=context,
                    zone=selected_zone,
                    recommendation=recommendation,
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown("### 💡 UrbanLeaf AI")

            st.success(answer)

            # ------------------------------------------------
            # RAG SOURCES
            # ------------------------------------------------

            if results:

                with st.expander(
                    "📚 View Retrieved Evidence"
                ):

                    for item in results:

                        st.markdown(
                            f"**📄 {item['source']}** "
                            f"· relevance {item['score']:.2f}"
                        )

                        preview = (
                            item["text"]
                            .replace("\n", " ")
                        )

                        if len(preview) > 500:
                            preview = (
                                preview[:500] + "..."
                            )

                        st.caption(preview)

            else:

                st.warning(
                    "No relevant knowledge was retrieved."
                )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.markdown("---")

            st.markdown(
                """
                **🧠 UrbanLeaf AI Pipeline**

                `🎤 Voice / 💬 Text`
                → `RAG`
                → `Trusted Knowledge`
                → `Gemini`
                → `🌱 AI Explanation`
                """
            )

    # --------------------------------------------------------
    # Example questions
    # --------------------------------------------------------

    examples = [
        "Why is this zone a high priority?",
        f"Why is {recommendation['species']} recommended here?",
        "What data supports this decision?",
        "What should the municipality do first?",
        "How can GIS and satellite data improve this?",
        "How can soil sensors improve tree selection?",
    ]

    example_question = st.selectbox(
    "💡 Try an example",
    examples,
    key="ai_copilot_example_question"
)

    question = st.text_input(
    "Ask UrbanLeaf AI",
    value=example_question,
    placeholder="Ask something about this zone...",
    key="ai_copilot_question"
)
    # --------------------------------------------------------
    # ASK AI
    # --------------------------------------------------------

    if st.button(
    "🤖 Ask UrbanLeaf AI",
    type="primary",
    width="stretch",
    key="ai_copilot_ask_button"
):

        with st.spinner(
            "Retrieving evidence and consulting UrbanLeaf AI..."
        ):

            # -----------------------------------------------
            # STEP 1 — RAG retrieval
            # -----------------------------------------------

            results = retrieve(
                question,
                top_k=3
            )

            context = "\n\n".join(
                item["text"]
                for item in results
            )

            # -----------------------------------------------
            # STEP 2 — Gemini LLM
            # -----------------------------------------------

            answer = generate_grounded_answer(
                question=question,
                context=context,
                zone=selected_zone,
                recommendation=recommendation,
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown("### 💡 UrbanLeaf AI")

        st.success(answer)

        # ----------------------------------------------------
        # RAG SOURCES
        # ----------------------------------------------------

        if results:

            with st.expander(
                "📚 View Retrieved Evidence"
            ):

                for item in results:

                    st.markdown(
                        f"**📄 {item['source']}** "
                        f"· relevance {item['score']:.2f}"
                    )

                    preview = (
                        item["text"]
                        .replace("\n", " ")
                    )

                    if len(preview) > 500:
                        preview = preview[:500] + "..."

                    st.caption(preview)

        # ----------------------------------------------------
        # ARCHITECTURE
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            """
            **🧠 UrbanLeaf AI Pipeline**

            `Question`
            → `RAG Retrieval`
            → `Trusted Knowledge`
            → `Gemini LLM`
            → `Grounded Explanation`
            """
        )

        st.caption(
            "UrbanLeaf separates deterministic environmental "
            "analysis from the LLM explanation layer."
        )

    # --------------------------------------------------------
    # Example questions
    # --------------------------------------------------------

    examples = [
        "Why is this zone a high priority?",
        f"Why is {recommendation['species']} recommended here?",
        "What data supports this decision?",
        "How can GIS and satellite data improve this?",
        "How can soil sensors improve tree selection?",
    ]

    example_question = st.selectbox(
        "💡 Try an example",
        examples
    )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    question = st.text_input(
        "Ask UrbanLeaf AI",
        value=example_question,
        placeholder="e.g. Why is this zone high priority?"
    )

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    if st.button(
        "🔎 Ask UrbanLeaf AI",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Searching UrbanLeaf knowledge base..."
        ):

            results = retrieve(
                question,
                top_k=3
            )

            context = "\n\n".join(
                item["text"]
                for item in results
            )

            answer = answer_from_context(
                question,
                context,
                selected_zone,
                recommendation
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown("### 💡 AI Explanation")

        st.success(answer)

        # ----------------------------------------------------
        # Retrieved sources
        # ----------------------------------------------------

        if results:

            st.markdown(
                "### 📚 Retrieved Knowledge"
            )

            for item in results:

                source_name = item["source"]

                relevance = item["score"]

                st.markdown(
                    f"**{source_name}** "
                    f"· relevance {relevance:.2f}"
                )

                preview = (
                    item["text"]
                    .replace("\n", " ")
                )

                preview = preview[:420]

                if len(item["text"]) > 420:
                    preview += "..."

                st.caption(preview)

        else:

            st.warning(
                "Knowledge base not found. "
                "Make sure the knowledge_base folder "
                "exists in the UrbanLeafAI project root."
            )

    # --------------------------------------------------------
    # RAG explanation
    # --------------------------------------------------------

    st.info(
        "🧠 RAG foundation: UrbanLeaf retrieves "
        "relevant information from the local "
        "knowledge base before generating the "
        "prototype explanation. The next stage "
        "will connect this retrieved context to "
        "a real LLM for conversational responses."
    )
# ============================================================
# WHAT-IF
# ============================================================
elif nav == "🔮 What-If Simulator":
    st.subheader(f"🔮 What-If Climate Simulator — {selected_zone['zone']}")
    trees=st.slider("🌳 Number of trees to plant",0,2000,1000,50)
    carbon=trees*0.060; green_inc=min(trees*0.0082,8.2); heat=min(trees/1000*14,14); people=int(trees*12.4)
    a,b,c,d=st.columns(4)
    with a: st.metric("CO₂ Capture",f"+{carbon:.1f} t/yr")
    with b: st.metric("Green Cover",f"+{green_inc:.1f}%")
    with c: st.metric("Heat-Risk Indicator",f"-{heat:.1f}%")
    with d: st.metric("People Potentially Benefited",f"{people:,}")
    chart=pd.DataFrame({"Metric":["CO₂ Capture","Green Cover"],"Current":[float(selected_zone['existing_trees'])*.060,float(selected_zone['green_cover'])],"Scenario":[float(selected_zone['existing_trees'])*.060+carbon,float(selected_zone['green_cover'])+green_inc]})
    long=chart.melt(id_vars="Metric",var_name="Scenario",value_name="Value")
    fig=px.bar(long,x="Metric",y="Value",color="Scenario",barmode="group",height=360)
    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,width="stretch")
    st.caption("Prototype scenario estimates for comparison, not certified environmental measurements.")

# ============================================================
# TREE LIBRARY
# ============================================================
elif nav == "🌳 Tree Library":
    st.subheader(f"🌳 Tree Intelligence — {selected_zone['zone']}")
    profiles=species_library(); scores=species_scores(selected_zone); rec=recommendation['species']
    st.success(f"🤖 AI match: **{rec}**")
    cols=st.columns(4)
    for col,(species,profile) in zip(cols,profiles.items()):
        with col:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            st.markdown(f"### {'⭐ ' if species==rec else ''}{species}")
            st.progress(int(scores[species]))
            st.caption(f"Suitability: **{scores[species]:.0f}/100**")
            st.write(f"**Best for:** {profile['best_for']}")
            st.write(f"**Space:** {profile['space']}")
            st.write(f"**CO₂ factor:** {profile['carbon']:.3f} t/tree/year")
            st.caption(profile['reason'])
            st.markdown("</div>",unsafe_allow_html=True)
    st.info("Suitability is a transparent prototype score using temperature, green-cover deficit, pollution and available planting space. Production deployment should calibrate with local ecology, soil, water and GIS data.")

# ============================================================
# REPORTS — VISUAL / DIAGRAM REPORT
# ============================================================
elif nav == "📄 Reports":
    st.subheader(f"📄 Visual Municipal Report — {selected_zone['zone']}")
    breakdown=get_priority_breakdown(selected_zone); focus=focus_areas(selected_zone); score=float(selected_zone['priority_score']); level,_=priority_class(score)

    st.markdown("<div class='report-box'><b>Designed for fast municipal retrieval:</b> the report uses a visual action-card format so an officer can identify the zone, urgency, cause, recommended tree, planting scale and expected scenario impact in seconds.</div>",unsafe_allow_html=True)
    st.write("")
    r1,r2,r3=st.columns(3)
    with r1: st.metric("Priority",f"{level} • {score:.0f}/100")
    with r2: st.metric("Action",f"{recommendation['trees']:,} trees")
    with r3: st.metric("Scenario CO₂",f"{recommendation['carbon_capture']:.1f} t/yr")

    aq = get_real_aq(selected_zone)
    if aq:
        st.markdown("### 🌍 Real Data Evidence")
        station = aq.get("nearest_station", "OpenAQ")
        distance = aq.get("distance_km")
        distance_text = f"{float(distance):.2f} km away" if distance is not None else "nearest matched station"
        aq_parts = []
        for key, label in [("pm25", "PM2.5"), ("pm10", "PM10"), ("no2", "NO₂")]:
            if key in aq:
                aq_parts.append(f"{label}: {float(aq[key]):.2f}")
        st.info(
            f"**OpenAQ:** {station} • {distance_text}"
            + (f" • {' • '.join(aq_parts)}" if aq_parts else "")
        )

    st.markdown("### 🧩 Report Diagram Preview")
    p1,p2,p3,p4,p5=st.columns(5)
    items=[("📍","WHERE",selected_zone['zone']),("🎯","WHY",f"{score:.0f}/100"),("🌳","WHAT",recommendation['species']),("📋","ACTION",f"{recommendation['trees']:,} trees"),("🌍","IMPACT",f"{recommendation['carbon_capture']:.1f} t/yr")]
    for col,(icon,title,value) in zip([p1,p2,p3,p4,p5],items):
        with col:
            st.markdown(f"<div class='card' style='text-align:center'><div style='font-size:24px'>{icon}</div><b>{title}</b><br/><span class='mini'>{value}</span></div>",unsafe_allow_html=True)

    pdf=make_visual_pdf(selected_zone,recommendation,breakdown,focus)
    if pdf:
        st.download_button("📥 Download Visual PDF Action Report",data=pdf,file_name=f"UrbanLeaf_AI_{selected_zone['zone'].replace(' ','_')}_Visual_Action_Report.pdf",mime="application/pdf",width="stretch")
    else:
        st.warning("ReportLab is not installed, so the visual PDF cannot be generated.")
    st.caption(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} • Prototype decision-support output")
