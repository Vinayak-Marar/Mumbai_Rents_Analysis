import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import yaml
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

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:

    try:
        reg = RandomForestRegressor(max_depth=None, max_features=None, n_estimators=50, n_jobs=-1, min_samples_leaf=1, min_samples_split=5)
        reg.fit(X_train, y_train.ravel())
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

def main():
    try:

        X_train = load_data('./data/transformed/X_train.csv')
        y_train = load_data('./data/transformed/y_train.csv')


        reg = train_model(X_train, y_train.values)
        
        save_model(reg, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()