# 🏙️ Mumbai Rent Prediction & Recommendation System

## 📌 Project Overview

This project is an **end-to-end machine learning system** for predicting apartment rents in Mumbai and recommending suitable rental apartments based on user preferences.

It combines:

* **Machine Learning** for rent prediction
* **Recommendation logic** for apartment matching
* **Data visualization** for locality-wise and area-wise insights
* **MLOps practices** using DVC and MLflow
* **FastAPI** for serving predictions via a web interface

The project is designed as a **production-style ML system**, not a standalone notebook demo.

---

## ✨ Key Features

* 🔮 **Rent Prediction** using a trained Random Forest model
* 🏠 **Apartment Recommendation System** based on budget, locality, and preferences
* 📊 **Locality-wise & area-wise analysis and visualizations**
* 🌐 **Web interface** (HTML + CSS)
* ⚙️ **FastAPI REST backend**
* 📈 **MLflow** for experiment and model tracking
* 📦 **DVC-managed pipeline** for data and artifacts
* 🐳 **Dockerized** application
* ☁️ Designed for **AWS (S3, ECR, EC2)** deployment

---

## 🧠 Machine Learning Details

* **Problem type**: Regression (rent prediction)
* **Algorithm used**: Random Forest Regressor
* **Model selection**: Chosen based on consistent and convincing performance
* **Experiment tracking**: MLflow

---

## 🧾 Prediction Inputs

The rent prediction endpoint accepts the following features:

* Bedrooms
* Built-up area (sqft)
* Bathrooms
* Balconies
* Facing
* Furnishing type
* Current floor
* Total floors
* Covered parking
* Open parking
* Locality (village)
* Lease type(s)
* Amenities
* Nearby facilities

---

## 🏷️ Recommendation Inputs

The recommendation system accepts:

* Locality (village)
* Budget
* Bedrooms (optional)
* Area preference (optional)
* Lease type(s)
* Amenities  (optional)
* Nearby facilities  (optional)

It returns apartments that best match the given constraints.

---

## 🌍 Geolocation Support

* Uses **OpenCage Geocoding API** to obtain **latitude and longitude** for Mumbai localities
* Enables locality-aware analysis and recommendation logic

---

## 🧑‍💻 Tech Stack

* **Language**: Python
* **Data & ML**: Pandas, NumPy, Scikit-learn
* **Visualization**: Matplotlib
* **Backend**: FastAPI
* **MLOps**: DVC, MLflow
* **Deployment**: Docker, AWS (S3, ECR, EC2 – previously deployed)
* **Frontend**: HTML, CSS

---

## 📂 Dataset

* **Source**: Custom dataset scraped using:

  * Common Crawl
  * Chrome-based web scraping extension
* **Domain**: Mumbai rental housing market

---

## 📁 Repository Structure

```
├── data/
├── models/
├── experiments/
├── src/
├── .github/workflows/
├── .venv/
├── dvc.yaml
├── dvc.lock
├── main.py
├── Dockerfile
├── requirements.txt
├── .gitignore
```

---

## 🔐 Data & Model Access Notice (IMPORTANT)

> Datasets and trained model artifacts are stored in a **private AWS S3 bucket** and are **not publicly accessible**.

Because of this:

* Large datasets and models are **intentionally excluded from GitHub**
* The **full DVC pipeline cannot be reproduced without S3 access**

This repository is meant to demonstrate:

* ML system design
* DVC-based pipeline management
* MLflow experiment tracking
* API integration and deployment strategy

---

## ▶️ Running the Project

### ⚠️ Execution Reality

Without access to the private S3 bucket:

* You **cannot** run the full pipeline end-to-end
* You **can** review all code, pipeline stages, and system design

This mirrors **real-world ML systems**, where data and models are not public.

---

### 🔹 Prerequisites

* Python **3.10+**
* DVC
* AWS credentials (required only for full execution)

```bash
pip install dvc[s3]
```

---

### 🔹 Setup (Code Review / Limited Mode)

```bash
git clone <your-repo-url>
cd Mumbai-Rent-Analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### 🔹 Full Execution (Requires Private S3 Access)

```bash
dvc repro
```

This step:

* Ingests data from S3
* Prepares features
* Ensures trained model artifacts are available

After successful execution:

```bash
uvicorn main:app --reload
```

---

## 🐳 Docker

```bash
docker build -t mumbai-rent-app .
docker run -p 8000:8000 mumbai-rent-app
```

---

## ☁️ Deployment Note

The application was previously deployed on **AWS EC2**, with Docker images stored in **ECR** and data managed via **S3**.
Cloud resources have since been removed, but the project structure remains deployment-ready.

---

## 👤 Author

**GitHub**: Vinayak Marar

---

## 📌 Final Note

This project prioritizes **correct ML engineering practices** over convenience:

* No large binaries in GitHub
* No public data leakage
* Reproducibility handled via DVC



