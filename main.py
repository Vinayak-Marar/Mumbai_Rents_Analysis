from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
import io, base64
import plotly.express as px
import os


FEATURES = ["builtup_area","rooms","furnish","bathrooms", "balcony",\
            "facing","gas_pipline","gated_community","swimming_pool","gym","intercom",\
             "power_backup","garden","sports","current_floor","total_floor","lease_type",\
             "covered_parking","open_parking","school/university","airport","bus_stop",\
             "railway","mall","metro_station","hospital","restaurant","latitude","longitude"]

AMENITIES = ['gas_pipeline', "gated_community", 'swimming_pool', "gym", "intercom", "power_backup", "garden", "sports"]

NEARBY = ["school", "airport","bus_stop", "railway", "mall", "metro_station","hospital", "restaurant"]


#import transformer and the model

model = joblib.load('models/model.pkl')
transformer = joblib.load('models/transformer.pkl')
df = pd.read_csv("data/visualization_data/data.csv")
quantiles = df["rent_density"].quantile([0, 0.33,0.5, 0.66, 0.85, 1]).round(6)

df["density_bin"] = pd.cut(
    df["rent_density"],
    bins=quantiles,
    labels=["Very_Low", "Low", "Medium", "High", "Very_High"],
    # include_lowest=True
)

df["density_bin"] = pd.Categorical(
    df["density_bin"],
    categories=["Very_Low", "Low", "Medium", "High", "Very_High"],
    ordered=True
)

color_map = {
    "Very Low":  "#deebf7",  # lightest
    "Low":       "#9ecae1",
    "Medium":    "#6baed6",
    "High":      "#3182bd",
    "Very High": "#08519c"   # darkest
}


# print(df)

bivariate_features = ["builtup_area", "bathrooms","furnish", "facing", "rooms", "balcony", "current_floor", "total_floor"]

def normalize_lease_type(x):
    # Handle missing values properly
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Bachelor Company Family"

    # If x is a list → convert cleanly
    if isinstance(x, list):
        parts = sorted(set(x))

    # If x is a string like "Bachelor Company"
    elif isinstance(x, str):
        parts = sorted(set(x.split()))

    # If it's something weird
    else:
        return "Bachelor Company Family"

    return " ".join(parts)


load_dotenv() 

KEY = os.getenv("opencage")
geocoder = OpenCageGeocode(KEY)

app = FastAPI()

def get_coordinates(village: str):
    result = geocoder.geocode(village)
    if result:
        lat = result[0]["geometry"]["lat"]
        lng = result[0]["geometry"]["lng"]
        return lat, lng
    return {"error": "Village not found"}

def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def get_bivariate_plot_base64(feature):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 10))

    # CASE 1: Numeric Feature → Bin → Mean Rent Bar Plot
    # if pd.api.types.is_numeric_dtype(df[feature]):
    if feature == "builtup_area":

        # Create quantile bins (automatically balanced)
        df["bin"] = pd.qcut(df[feature], q=10, duplicates="drop")

        # Compute mean rent per bin
        bin_means = df.groupby("bin")["rent"].mean()

        # Plot
        bin_means = bin_means.sort_index()
        bin_means.plot(kind="bar", ax=ax)

        ax.set_title(f"Mean Rent vs {feature} (Binned)", color="white")
        ax.set_xlabel(f"{feature} (binned ranges)", color="white")
        ax.set_ylabel("Mean Rent", color="white")
        plt.xticks(rotation=45, ha="right", color="white")

        # Cleanup
        df.drop(columns=["bin"], inplace=True)

    # CASE 2: Categorical Feature → Mean Rent Bar Plot
    else:
        means = df.groupby(feature)["rent"].mean().sort_values()

        means = means.sort_index()
        means.plot(kind="bar", ax=ax)

        ax.set_title(f"Mean Rent vs {feature}", color="white")
        ax.set_xlabel(feature, color="white")
        ax.set_ylabel("Mean Rent", color="white")
        plt.xticks(rotation=45, ha="right", color="white")

    # Return base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()

    return f"data:image/png;base64,{encoded}"




templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
def index(request: Request):
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
    try:
        lat, long = get_coordinates(village=village)

    except Exception as e:
        lat, long = None, None
        return JSONResponse(content={
            "error": f"Could not fetch coordinates for '{village}'. Try a nearby landmark or different spelling."
        })

    basic_inputs = [ builtup_area, rooms, furnish_type, bathroom, balconies,facing]
    amenities_list = [1 if x.lower() in [y.lower() for y in amenities] else 0 for x in AMENITIES]
    lease_type = normalize_lease_type(lease)
    floors = [current_floor, total_floor, lease_type]
    
    nearby_list = [1 if x.lower() in [y.lower() for y in nearby] else 0 for x in NEARBY]
    lat_long = [lat, long]

    input_data = basic_inputs + amenities_list + floors + [covered_parking, open_parking] + nearby_list + lat_long

    print(lat_long)
    # Make prediction


    input_data = np.array(input_data).reshape(1,-1)
    input_df = pd.DataFrame(input_data, columns=FEATURES)

    transformer.set_output(transform="pandas")
    transformed_data = transformer.transform(input_df)

    actual_df = transformed_data
    predicted_rent = model.predict(actual_df)[0]
    # predicted_rent = 10000

    # Return HTML page with result

    return JSONResponse(content={"result": round(float(predicted_rent), -2)})


@app.get("/visualization", response_class=HTMLResponse)
def visualization_page(request: Request):
    # Plotly map
    fig_map = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="density_bin",
        # color_discrete_map=color_map,
        hover_name="sector_area",
        hover_data=["rent", "rooms", "builtup_area"],
        zoom=10,
        mapbox_style="carto-positron",
            color_discrete_map=color_map,
        category_orders={"density_bin": ["Very_Low", "Low", "Medium", "High", "Very_High"] },
        title="Rent Density (5-Level Categorized)"
    )
    fig_map.update_layout(dragmode="zoom", height=650)
    
    map_html = fig_map.to_html(full_html=False, config={"scrollZoom": True})

    # Initial bivariate image (first feature)
    initial_feature = bivariate_features[0]
    bivar_image = get_bivariate_plot_base64(initial_feature)

    return templates.TemplateResponse("visualization.html", {
        "request": request,
        "map_plot": map_html,
        "features": bivariate_features,
        "bivar_image": bivar_image,
        "selected_feature": initial_feature
    })

# AJAX endpoint for bivariate plot
@app.post("/get_bivariate_plot")
def get_bivariate_plot(feature: str = Form(...)):
    img_base64 = get_bivariate_plot_base64(feature)
    return JSONResponse(content={"img": img_base64})

@app.get("/recommendation")
def recommendation_form(request: Request):
    # Show the form
    return templates.TemplateResponse("recommendation.html", {"request": request})


@app.post("/recommendation", response_class=HTMLResponse)
def recommendation_page(
    request: Request,

    # required inputs
    village: str = Form(...),
    budget: float = Form(...),

    # lease is checkbox → list
    lease: list[str] | None = Form(None),

    # optional inputs
    rooms: str = Form(None),
    area: str = Form(None),
    amenities: list[str] | None = Form(None),
    nearby: list[str] | None = Form(None)
):

    # ---------- SAFE CAST ----------
    def safe_float(x):
        try:
            return float(x)
        except:
            return None

    def safe_int(x):
        try:
            return int(x)
        except:
            return None

    rooms = safe_int(rooms)
    area = safe_float(area)

    # ---------- GEOCODE VILLAGE INTO LAT/LON ----------
    # geo = GEOCODER.geocode(village)

    user_lat, user_lon = get_coordinates(village=village)

    # if not geo:
    #     user_lat, user_lon = None, None
    # else:
    #     user_lat = geo[0]["geometry"]["lat"]
    #     user_lon = geo[0]["geometry"]["lng"]

    # ---------- USER FEATURE VECTOR ----------
    user_vec = {
        "budget": budget,
        "rooms": rooms or 0,
        "area": area or 0,
        "lease_family": 1 if lease and "Family" in lease else 0,
        "lease_bachelors": 1 if lease and "Bachelors" in lease else 0,
        "lease_company": 1 if lease and "Company" in lease else 0,
    }

    # Amenities
    amen_list = [
        "gas_pipeline","gated_community","swimming_pool","garden",
        "sports","gym","intercom","power_backup"
    ]
    for a in amen_list:
        user_vec["amen_" + a] = 1 if (amenities and a in amenities) else 0

    # Nearby facilities
    near_list = [
        "school","bus_stop","railway_station","mall","metro_station",
        "airport","hospital","restaurant"
    ]
    for n in near_list:
        user_vec["near_" + n] = 1 if (nearby and n in nearby) else 0

    user_vector = np.array(list(user_vec.values())).reshape(1, -1)

    # ---------- BUILD PROPERTY VECTORS FROM DATA ----------
    property_vectors = []
    df_indices = []

    for idx, row in df.iterrows():
        vec = [
            row.get("rent", 0),
            row.get("rooms", 0),
            row.get("builtup_area", 0),
            1 if row.get("lease_type") == "Family" else 0,
            1 if row.get("lease_type") == "Bachelors" else 0,
            1 if row.get("lease_type") == "Company" else 0,
        ]

        # Amenity columns exist?
        for a in amen_list:
            vec.append(1 if row.get(a) == 1 else 0)

        for n in near_list:
            vec.append(1 if row.get(n) == 1 else 0)

        property_vectors.append(vec)
        df_indices.append(idx)

    property_vectors = np.array(property_vectors)

    # ---------- COSINE SIMILARITY ----------
    sims = cosine_similarity(user_vector, property_vectors)[0]

    df["similarity"] = sims

    # ---------- FILTER BY REQUIRED ----------
    recs = df[
    (df['latitude'].between(user_lat - 0.05, user_lat + 0.05)) &
    (df['longitude'].between(user_lon - 0.05, user_lon + 0.05)) &
    (df['rent'] <= budget+ 0.15*budget)
    ].sort_values("similarity", ascending=False).head(10)

    if not recs.empty:
        fig = px.scatter_mapbox(
            recs,
            lat="latitude",
            lon="longitude",
            hover_name="sector_area",
            hover_data=["rent", "rooms", "builtup_area"],
            zoom=12,
            height=500
        )
        fig.update_layout(mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        map_html = fig.to_html(full_html=False, config={"scrollZoom": True})
    else:
        map_html = "<p>No recommendations found for your criteria.</p>"

    # ---------- RENDER RESULT ----------
    return templates.TemplateResponse(
        "recommendation_results.html",
        {
            "request": request,
            "recommendation": recs.to_dict(orient="records"),
            "map_plot": map_html
        }
    )
