import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

import os
import yaml
import joblib
import json

from src.logger.logger import logging


def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, **params) -> RandomForestRegressor:

    try:
        reg = RandomForestRegressor(max_depth=None, max_features=None, n_estimators=500, n_jobs=-1, min_samples_leaf=1, min_samples_split=5)
        reg.fit(X_train, y_train)
        logging.info('Model training completed')
        return reg
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:

    try:
        joblib.dump(model, file_path)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def save_run_info(run_id: str, model_path: str) -> None:
    """Save MLflow run info for model registration."""
    info = {
        "run_id": run_id,
        "model_path": model_path
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/experiment_info.json", "w") as f:
        json.dump(info, f, indent=4)
    logging.info("Experiment info saved")

def main():
    try:

        X_train = load_data('./data/transformed/X_train.csv')
        y_train = load_data('./data/transformed/y_train.csv')

        # Convert to numpy arrays
        X_train = X_train.values
        y_train = y_train.values

        # MLflow experiment name
        mlflow.set_experiment("model_training")

        with mlflow.start_run() as run:

            # Train model
            reg = train_model(X_train, y_train)

            # Log model parameters to MLflow
            if hasattr(reg, 'get_params'):
                params = reg.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(reg, artifact_path = "model")
            
            # Save model locally
            save_model(reg, "models/model.pkl")

            # Save run data for registry
            save_run_info(run.info.run_id, "model")

            logging.info(f"Model training MLflow run_id: {run.info.run_id}")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        print(f"Error: {e}")
        

if __name__ == '__main__':
    main()