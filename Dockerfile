# Base Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only what we need
COPY data/ /app/data/
COPY main.py /app/main.py
COPY models/model.pkl /app/models/model.pkl
COPY models/transformer.pkl /app/models/transformer.pkl
COPY templates/ /app/templates/
COPY static/ /app/static/

# Install dependencies
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
