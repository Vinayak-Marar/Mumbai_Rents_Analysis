from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode


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
bivariate_features = ["builtup_area", "bathroom", "rooms", "balconies", "current_floor", "total_floor"]

def normalize_lease_type(x):
    if pd.isna(x):
        return 'Bachelor Company Family'
    # Split by spaces, remove empty, unique, and sort alphabetically
    parts = sorted(set(x))
    return ' '.join(parts)


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


def get_bivariate_plot_base64(feature: str):
    plt.figure(figsize=(8,5))
    plt.scatter(df[feature], df["rent"], c='blue', alpha=0.6)
    plt.xlabel(feature.replace("_"," ").title())
    plt.ylabel("Rent")
    plt.title(f"Rent vs {feature.replace('_',' ').title()}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
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
    lat, long = get_coordinates(village=village)

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
    fig_map = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        z="rent_density",
        radius=17,
        center=dict(lat=19.0760, lon=72.8777),
        zoom=10,
        mapbox_style="carto-positron",
        hover_name="sector_area",
        hover_data=["rent", "rooms", "builtup_area"]
    )
    fig_map.update_layout(dragmode="zoom")
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
    village: str = None,
    budget: float = None,
    lease: str = None,
    area: float = None,
    rooms: int = None,
    bedrooms: int = None
):
    # Filter dataset based on user inputs
    if village:  # only filter if village input is provided
        filtered_df = df[df['area'].astype(str).str.contains(village, case=False, na=False)]
    else:
        filtered_df = df.copy()
        
    if budget:
        filtered_df = filtered_df[filtered_df['rent'] <= budget]
    if area:
        filtered_df = filtered_df[filtered_df['builtup_area'] >= area]
    if rooms:
        filtered_df = filtered_df[filtered_df['rooms'] >= rooms]
    if bedrooms and 'bedrooms' in df.columns:
        filtered_df = filtered_df[filtered_df['bedrooms'] >= bedrooms]
    if lease:
        # assuming you have a column 'lease_type' in lowercase
        filtered_df = filtered_df[filtered_df['lease_type'].str.lower() == lease.lower()]

    # Get top 5 recommendations by rent (closest to budget or cheapest)
    recommendation = filtered_df.nsmallest(5, 'rent')

    # Plotly map with markers
    if not recommendation.empty:
        fig = px.scatter_mapbox(
            recommendation,
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

    return templates.TemplateResponse("recommendation_results.html", {
        "request": request,
        "recommendation": recommendation.to_dict(orient="records"),
        "map_plot": map_html
    })