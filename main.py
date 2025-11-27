from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
FEATURES = [
    "builtup_area", "rooms", "furnish", "bathrooms", "balcony",
    "facing", "gas_pipline", "gated_community", "swimming_pool",
    "gym", "intercom", "power_backup", "garden", "sports",
    "current_floor", "total_floor", "lease_type",
    "covered_parking", "open_parking",
    "school/university", "airport", "bus_stop", "railway",
    "mall", "metro_station", "hospital", "restaurant",
    "latitude", "longitude"
]

AMENITIES = [
    'gas_pipeline', "gated_community", 'swimming_pool', "gym",
    "intercom", "power_backup", "garden", "sports"
]

NEARBY = [
    "school", "airport", "bus_stop", "railway",
    "mall", "metro_station", "hospital", "restaurant"
]

BIVAR_FEATURES = [
    "builtup_area", "bathrooms", "furnish", "facing",
    "rooms", "balcony", "current_floor"
]


# ---------------------------------------------
# LOAD MODELS + DATA
# ---------------------------------------------
model = joblib.load('models/model.pkl')
transformer = joblib.load('models/transformer.pkl')
df = pd.read_csv("data/visualization_data/data.csv")

quantiles = df["rent_density"].quantile([0, 0.33, 0.5, 0.66, 0.85, 1]).round(6)

df["density_bin"] = pd.cut(
    df["rent_density"],
    bins=quantiles,
    labels=["Very_Low", "Low", "Medium", "High", "Very_High"]
)

df["density_bin"] = pd.Categorical(
    df["density_bin"],
    categories=["Very_Low", "Low", "Medium", "High", "Very_High"],
    ordered=True
)

COLOR_MAP = {
    "Very Low": "#deebf7", "Low": "#9ecae1", "Medium": "#6baed6",
    "High": "#3182bd", "Very High": "#08519c"
}


# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def normalize_lease_type(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Bachelor Company Family"

    if isinstance(x, list):
        parts = sorted(set(x))

    elif isinstance(x, str):
        parts = sorted(set(x.split()))

    else:
        return "Bachelor Company Family"

    return " ".join(parts)


load_dotenv()
GEOKEY = os.getenv("opencage")
geocoder = OpenCageGeocode(GEOKEY)


def get_coordinates(village: str):
    result = geocoder.geocode(village)
    if result:
        g = result[0]["geometry"]
        return g["lat"], g["lng"]
    return None, None


def get_bivariate_plot_base64(feature):

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 10))

    if feature == "builtup_area":
        df["bin"] = pd.qcut(df[feature], q=10, duplicates="drop")
        bin_means = df.groupby("bin")["rent"].mean().sort_index()
        bin_means.plot(kind="bar", ax=ax)

        df.drop(columns=["bin"], inplace=True)

    else:
        means = df.groupby(feature)["rent"].mean().sort_index()
        means.plot(kind="bar", ax=ax)

    ax.set_title(f"Mean Rent vs {feature}", color="white")
    ax.set_xlabel(feature, color="white")
    ax.set_ylabel("Mean Rent", color="white")
    plt.xticks(rotation=45, ha="right", color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()

    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------
# ROUTES
# ---------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/prediction", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})


@app.post("/prediction", response_class=HTMLResponse)
async def predict_rent(
    request: Request,
    rooms: int = Form(...),
    builtup_area: int = Form(...),
    bathroom: int = Form(...),
    balconies: int = Form(0),
    facing: str = Form(...),
    furnish_type: str = Form(...),
    current_floor: int = Form(...),
    total_floor: int = Form(...),
    covered_parking: int = Form(...),
    open_parking: int = Form(...),
    village: str = Form(...),
    lease: list[str] = Form([]),
    amenities: list[str] = Form([]),
    nearby: list[str] = Form([]),
):
    lat, lon = get_coordinates(village)

    if lat is None:
        return JSONResponse({"error": f"Coordinates not found for '{village}'"})

    basic_inputs = [builtup_area, rooms, furnish_type, bathroom, balconies, facing]
    amenities_list = [1 if a.lower() in amenities else 0 for a in AMENITIES]

    lease_type = normalize_lease_type(lease)
    floors = [current_floor, total_floor, lease_type]

    nearby_list = [1 if n.lower() in nearby else 0 for n in NEARBY]
    latlon = [lat, lon]

    input_data = (
        basic_inputs +
        amenities_list +
        floors +
        [covered_parking, open_parking] +
        nearby_list +
        latlon
    )

    input_df = pd.DataFrame([input_data], columns=FEATURES)

    transformer.set_output(transform="pandas")
    tdata = transformer.transform(input_df)

    predicted = model.predict(tdata)[0]

    return JSONResponse({"result": round(float(predicted), -2)})


@app.get("/visualization", response_class=HTMLResponse)
def visualization_page(request: Request):

    fig_map = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="density_bin",
        hover_name="sector_area",
        hover_data=["rent", "rooms", "builtup_area"],
        zoom=10,
        mapbox_style="carto-positron",
        color_discrete_map=COLOR_MAP,
        category_orders={"density_bin": ["Very_Low", "Low", "Medium", "High", "Very_High"]},
        height=650
    )

    fig_map.update_layout(dragmode="zoom")
    map_html = fig_map.to_html(full_html=False, config={"scrollZoom": True})

    feature = BIVAR_FEATURES[0]
    bivar_img = get_bivariate_plot_base64(feature)

    return templates.TemplateResponse("visualization.html", {
        "request": request,
        "map_plot": map_html,
        "features": BIVAR_FEATURES,
        "bivar_image": bivar_img,
        "selected_feature": feature
    })


@app.post("/get_bivariate_plot")
def get_bivariate_plot(feature: str = Form(...)):
    img = get_bivariate_plot_base64(feature)
    return JSONResponse({"img": img})


# ---------------------------------------------------
# RECOMMENDATION SYSTEM
# ---------------------------------------------------
@app.get("/recommendation")
def recommendation_form(request: Request):
    return templates.TemplateResponse("recommendation.html", {"request": request})


@app.post("/recommendation")
def recommendation_page(
    request: Request,
    village: str = Form(...),
    budget: float = Form(...),
    lease: list[str] = Form(None),
    rooms: str = Form(None),
    area: str = Form(None),
    amenities: list[str] = Form(None),
    nearby: list[str] = Form(None)
):

    def safe_float(x): 
        try: return float(x)
        except: return None

    def safe_int(x):
        try: return int(x)
        except: return None

    rooms = safe_int(rooms)
    area = safe_float(area)

    user_lat, user_lon = get_coordinates(village)

    user_vec = {
        "budget": budget,
        "rooms": rooms or 0,
        "area": area or 0,
        "lease_family": 1 if lease and "Family" in lease else 0,
        "lease_bachelors": 1 if lease and "Bachelors" in lease else 0,
        "lease_company": 1 if lease and "Company" in lease else 0,
    }

    amen_cols = [
        "gas_pipeline", "gated_community", "swimming_pool",
        "garden", "sports", "gym", "intercom", "power_backup"
    ]

    for a in amen_cols:
        user_vec["amen_" + a] = 1 if (amenities and a in amenities) else 0

    near_cols = [
        "school", "bus_stop", "railway_station", "mall",
        "metro_station", "airport", "hospital", "restaurant"
    ]

    for n in near_cols:
        user_vec["near_" + n] = 1 if (nearby and n in nearby) else 0

    user_vector = np.array(list(user_vec.values())).reshape(1, -1)

    # Build property vectors
    property_vectors = []
    for _, row in df.iterrows():
        vec = [
            row.get("rent", 0),
            row.get("rooms", 0),
            row.get("builtup_area", 0),
            1 if row.get("lease_type") == "Family" else 0,
            1 if row.get("lease_type") == "Bachelors" else 0,
            1 if row.get("lease_type") == "Company" else 0,
        ]

        for a in amen_cols:
            vec.append(1 if row.get(a) else 0)

        for n in near_cols:
            vec.append(1 if row.get(n) else 0)

        property_vectors.append(vec)

    property_vectors = np.array(property_vectors)

    sims = cosine_similarity(user_vector, property_vectors)[0]
    df["similarity"] = sims

    recs = df[
        (df['latitude'].between(user_lat - 0.05, user_lat + 0.05)) &
        (df['longitude'].between(user_lon - 0.05, user_lon + 0.05)) &
        (df['rent'] <= budget * 1.15)
    ].sort_values("similarity", ascending=False).head(10)

    if not recs.empty:
        fig = px.scatter_mapbox(
            recs,
            lat="latitude", lon="longitude",
            hover_name="sector_area",
            hover_data=["rent", "rooms", "builtup_area"],
            zoom=12, height=500,
            mapbox_style="carto-positron"
        )
        fig.update_layout(mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        map_html = fig.to_html(full_html=False, config={"scrollZoom": True})
    else:
        map_html = "<p>No recommendations found.</p>"

    return templates.TemplateResponse(
        "recommendation_results.html",
        {"request": request, "recommendation": recs.to_dict("records"), "map_plot": map_html}
    )
