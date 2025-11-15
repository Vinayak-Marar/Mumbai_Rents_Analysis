from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
# import shap
import matplotlib.pyplot as plt
import io, base64
import plotly.express as px
import os

app = FastAPI()

df = pd.read_csv("full_df.csv")
bivariate_features = ["builtup_area", "bathroom", "rooms", "balconies", "current_floor", "total_floor"]

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


model = joblib.load("model.pkl")
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
    village: str = Form(...),
    amenities: list[str] = Form([]),  # For multiple checkboxes
    nearby: list[str] = Form([]),     # For multiple checkboxes
):
    # Prepare input features for your model
    # You need to transform categorical features like 'facing', 'furnish_type', 'village', 'amenities', 'nearby' properly
    # Here is a simple example if your model expects numeric inputs only:
    input_data = [rooms, builtup_area, bathroom, balconies, current_floor, total_floor, village, amenities, nearby]
    
    # Make prediction


    #predicted_rent = model.predict(input_data)[0]
    predicted_rent = 10000

    print(input_data)
    # Return HTML page with result

    return JSONResponse(content={"result": round(float(predicted_rent), 2)})
    # return templates.TemplateResponse("prediction.html", {
    #     "request": request,
    #     "result": round(predicted_rent, 2)
    # })


# @app.get("/visualize", response_class=HTMLResponse)
# def visualize(request: Request):
#     # Example visualization
#     plt.figure()
#     plt.bar(["Age", "BMI", "Glucose"], [50, 25, 75])
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.getvalue()).decode()
#     return templates.TemplateResponse("visualize.html", {"request": request, "plot": img_base64})

@app.get("/visualization", response_class=HTMLResponse)
def visualization_page(request: Request):
    # Plotly map
    fig_map = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        z="rent",
        radius=10,
        center=dict(lat=19.0760, lon=72.8777),
        zoom=10,
        mapbox_style="carto-positron",
        hover_name="sector area",
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
            hover_name="sector area",
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