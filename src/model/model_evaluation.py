import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import mlflow
import mlflow.sklearn

import os
import json
from datetime import datetime

from src.logger.logger import logging

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# dagshub.init(repo_owner='vikashdas770', repo_name='YT-Capstone-Project', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        model = joblib.load(file_path)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
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

def evaluate_model(reg, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = reg.predict(X_test)
        # y_pred_proba = clf.predict_proba(X_test)[:, 1]

        r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(y_test, y_pred)


        metrics_dict = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)   
#     try:
#         model_info = {'run_id': run_id, 'model_path': model_path}
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logging.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logging.error('Error occurred while saving the model info: %s', e)
#         raise

def main():
    mlflow.set_experiment("model-evaluation")
    with mlflow.start_run(run_name=f"RandomForest_{datetime.now():%Y%m%d_%H%M%S}") as run:  
        try:
            reg = load_model('./models/model.pkl')
            X_test = load_data('./data/transformed/X_test.csv')
            y_test = load_data('./data/transformed/y_test.csv')
            

            metrics = evaluate_model(reg, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            

            # Save model info
            # save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()