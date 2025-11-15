import pandas as pd

import os
import yaml
import joblib


from src.logger import logging
from src.utils.utils import rename_columns, create_new_df




def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logging.debug('Parameters retrieved from %s', params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
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
            data = data[data['rooms'] != '']
            logging.info('Removed corrupted data')

        except  Exception as e:
            logging.error('Unexpected error occurred while removing corrupted rows: %s', e)
            raise
        
        # Deleting duplicated rows
        data = data.drop_duplicates()

        # fixing an outlier
        try:
            data.loc[data['rooms'] == 'Studio Flat for Rent', 'rooms'] = '1 R Independent House for Rent'
            data.loc[data['furnish'] == '', 'furnish'] = 'Unfurnished'
            data.loc[data['builtup_area'] == 'OVERVIEW\nEXPLORE NEIGHBOURHOOD', 'builtup_area'] = '400'
            logging.debug("fixing a Input errors ")
        except:
            pass

        # creating new rooms list

        df = create_new_df(data)
   


        base_features = ['Pclass', 'Sex_enc', 'Age', 'Fare', 'Embarked_enc', 'SibSp', 'Parch']

        train_df = train_data.copy()
        test_df = test_data.copy()

        # engineered features BEFORE scaling
        train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
        test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

        final_features = base_features + ['FamilySize']

        transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        X_train = pd.DataFrame(
            transformer.fit_transform(train_df[final_features]),
            columns=final_features
        )
        X_test = pd.DataFrame(
            transformer.transform(test_df[final_features]),
            columns=final_features
        )

        # attach target
        X_train['Survived'] = train_df['Survived'].values
        X_test['Survived'] = test_df['Survived'].values

        logging.info("Data transformation completed.")
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
    with open(file_path, 'wb') as file:
        joblib.dump(model, file)
    logging.info("Model saved to %s", file_path)


if __name__ == '__main__':
    train = load_data('./data/processed/train.csv')
    test = load_data('./data/processed/test.csv')
    X_train, X_test, transformer = transform_data(train, test)
    save_data(X_train, './data/transformed/train.csv')
    save_data(X_test, './data/transformed/test.csv')
    save_model(transformer, './models/transformer.pkl')
