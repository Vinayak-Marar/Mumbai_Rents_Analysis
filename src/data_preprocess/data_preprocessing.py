import pandas as pd
import numpy as np

import os
import yaml
import joblib


from src.logger.logger import logging
from src.utils.utils import rename_columns, create_new_df




def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logging.debug('Parameters retrieved from %s', params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    logging.info('Data loaded from %s', file_path)
    return df


def transform_data(data: pd.DataFrame) -> None:
    """
    Create engineered features first (FamilySize), then impute/scale.
    Returns: (train_df, test_df, transformer)
    """
    try:
        logging.info("Transforming data...")

        # Renaming columns
        data = rename_columns(data)

        # Removing corrupted rows
        try:
            # data['rooms'] = data['rooms'].astype(str)
            data = data.dropna(subset=['rooms'])
            logging.info('Removed Missing rooms data')

        except  Exception as e:
            logging.error('Unexpected error occurred while removing missing rooms rows: %s', e)
            raise
        
        # Deleting duplicated rows
        data = data.drop_duplicates()
        

        # fixing an outlier manually
        try:
            data.loc[data['rooms'] == 'Studio Flat for Rent', 'rooms'] = '1 R Independent House for Rent'
            data.loc[data['furnish'] == '', 'furnish'] = 'Unfurnished'
            data.loc[data['builtup_area'] == 'OVERVIEW\nEXPLORE NEIGHBOURHOOD', 'builtup_area'] = '400'
            logging.debug("fixing a Input errors ")
        except:
            pass

        # creating new rooms list

        data = data.reset_index(drop=True)
        data = data.astype(str)

        df = create_new_df(data)

        logging.info("Data Preprocessing completed.")
        return df

    except Exception:
        logging.exception("Unexpected error during preprocessing the data")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info('Data saved to %s', file_path)


# def save_model(model, file_path: str) -> None:
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     with open(file_path, 'wb') as file:
#         joblib.dump(model, file)
#     logging.info("Model saved to %s", file_path)


if __name__ == '__main__':
    df = load_data('./data/ingested/data.xlsx')
    # test = load_data('./data/processed/test.csv')
    df = transform_data(df)
    save_data(df, './data/processed/data.csv')
    # save_data(X_test, './data/transformed/test.csv')
    # save_model(transformer, './models/transformer.pkl')
