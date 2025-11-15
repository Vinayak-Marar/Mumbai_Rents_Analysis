import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



import os
import yaml
import json
import csv
# import logging

from src.logger.logger import logging
from src.data_preprocess.data_validation import RawDataValidation

pd.set_option('future.no_silent_downcasting', True)


def load_params(params_path: str) -> dict:

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)


def load_data(data_path: str) -> json:

    try:
        with open(data_path, 'r') as f:
            df_json = json.load(f)
        logging.info('Data loaded from %s', data_path)
        return df_json
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the Json file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def convert_to_csv(df_json: json) -> pd.DataFrame:

    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("Converting json to csv ...")

        df = pd.DataFrame(df_json)
                
        logging.info('Converted to CSV')
        return df
    
    except Exception as e:
        logging.error('Unexpected error during conversion: %s', e)
        raise


def save_data(data: pd.DataFrame, data_path: str) -> None:
  
    try:
        logging.info('Saving the data ')
        raw_data_path = os.path.join(data_path, 'ingested')
        os.makedirs(raw_data_path, exist_ok=True)
        data.to_csv(os.path.join(raw_data_path, "data.csv"), index=False, quoting = csv.QUOTE_ALL, escapechar="\\")
        # test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False,  quoting = csv.QUOTE_ALL, escapechar="\\")
        logging.info('Data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        logging.info("Data Ingestion Started")
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        print(f"root_dir: {root_dir}")
        path = os.path.join(root_dir, "data", "raw_data.json")
        
        
        df_json = load_data(data_path=path)
        converted_data = convert_to_csv(df_json)

        validator = RawDataValidation(converted_data)
        final_df = validator.run_all_checks()


        # train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(final_df, data_path='./data')

        logging.info("Data Ingestion Completed")
        
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()