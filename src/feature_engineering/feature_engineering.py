import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

import os

from src.logger.logger import logging
from src.utils.utils import get_lat_long


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logging.debug('Parameters retrieved from %s', params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info('Data loaded from %s', file_path)
    return df


def creating_new_features(data: pd.DataFrame, lat_long: pd.DataFrame) -> pd.DataFrame:

    try:
        logging.info("Adding new features...")

        try:

            data['area'] = [x.split(',')[-2].strip() for x in data.address]

            def get_sector(x):
                try:
                    return x.split(',')[-3].strip()
                except:
                    return ""
                
            data['sector'] = [get_sector(x) for x in data.address]

            data['sector_area'] = [x + " " + y for x,y in zip(data.sector,data.area)]

            logging.info("Created area,sector and sector_area features")

        except Exception as e:
            logging.error('Unexpected error occurred while creating new features (area, sector, sector_area): %s', e)
            raise

        try:
            latitude, longitude = get_lat_long(data, lat_long=lat_long)
            data['latitude'] = latitude
            data['longitude'] = longitude
            logging.info("Created latitude and longitude")

        except Exception as e:
            logging.error('Unexpected error occurred while creating latitude and longitude: %s', e)
            raise

        return data

    except Exception:
        logging.exception("Unexpected error during transform_data")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info('Data saved to %s', file_path)


# def save_model(model, file_path: str) -> None:
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     with open(file_path, 'wb') as file:
#         pickle.dump(model, file)
#     logging.info("Model saved to %s", file_path)


if __name__ == '__main__':
    test_size = 0.2
    data = load_data('./data/processed/data.csv')
    lat_long = pd.read_excel('./data/lat_long.xlsx')
    # test = load_data('./data/processed/test.csv')
    data = creating_new_features(data, lat_long)
    X = data.drop('rent', axis = 1)
    y = data['rent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    save_data(X_train, './data/interim/X_train.csv')
    save_data(X_test, './data/interim/X_test.csv')
    save_data(y_train, './data/interim/y_train.csv')
    save_data(y_test, './data/interim/y_test.csv')

    logging.info("Feature engineering completed")
    # save_model(transformer, './models/transformer.pkl')
