from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
# import shap
import matplotlib.pyplot as plt
import io, base64

app = FastAPI()

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
def predict_result(request: Request, age: float = Form(...), bmi: float = Form(...), glucose: float = Form(...)):
    data = np.array([[age, bmi, glucose]])
    prediction = model.predict(data)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return templates.TemplateResponse("prediction.html", {"request": request, "result": result})


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


print("hi")