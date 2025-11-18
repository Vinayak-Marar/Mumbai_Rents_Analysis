import pandas as pd
import numpy as np
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import joblib

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


def transform_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:

    try:
        logging.info("Transforming the data")

        final_features = ["builtup_area","rooms","furnish","bathrooms", "balcony",\
                          "facing","gas_pipline","gated_community","swimming_pool","gym","intercom",\
                          "power_backup","garden","sports","current_floor","total_floor","lease_type",\
                          "covered_parking","open_parking","school/university","airport","bus_stop",\
                          "railway","mall","metro_station","hospital","restaurant","latitude","longitude"]
        
        categorical_features = ["facing", "lease_type"]
        ordinal_features = ["furnish"]
        furnish_order=[["Unfurnished", "Semi Furnished", "Fully Furnished"]]

        try:
            X_train = X_train[final_features]
            X_test = X_test[final_features]

            print(X_train.iloc[0, :].values)

            logging.info("Filtered the data")

        except Exception as e:
            logging.error('Unexpected error occurred while filtering the data: %s', e)
            raise

        try:
            transformer = ColumnTransformer(
                transformers=[
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                    ("ord", OrdinalEncoder(categories=furnish_order,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1),
                    ordinal_features),
                ],
                remainder="passthrough"   
                )
            
            transformer.set_output(transform="pandas")

            X_train= transformer.fit_transform(X_train)

            X_test = transformer.transform(X_test)

        except Exception as e:
            logging.error('Unexpected error occurred while transforming the data: %s', e)
            raise
        

        return X_train, X_test, transformer

    except Exception:
        logging.exception("Unexpected error during transform_data")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info('Data saved to %s', file_path)


def save_model(model, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    logging.info("Model saved to %s", file_path)


if __name__ == '__main__':

    X_train =  load_data( './data/interim/X_train.csv')
    X_test  = load_data('./data/interim/X_test.csv')
    y_train = load_data( './data/interim/y_train.csv')
    y_test = load_data( './data/interim/y_test.csv')

    print(X_train.shape, X_test.shape)
    # test = load_data('./data/processed/test.csv')
    X_train, X_test, transformer = transform_data(X_train ,X_test)


    save_data(X_train, './data/transformed/X_train.csv')
    save_data(X_test, './data/transformed/X_test.csv')
    save_data(y_train, './data/transformed/y_train.csv')
    save_data(y_test, './data/transformed/y_test.csv')

    save_model(transformer, './models/transformer.pkl')

    logging.info("Data Transform Completed")

