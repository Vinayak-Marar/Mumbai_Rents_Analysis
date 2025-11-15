import os
# import json
import pandas as pd
import logging

class RawDataValidation:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # These MUST be the raw column names (as they appear in the file)
        self.expected_columns = [
            "css-1hidc9c",
            "css-gkudnx",
            "T_textContainerStyle",
            "css-1ty5xzi",
            "css-10rvbm3",
            "_26jk66",
            "T_overviewStyle",
            "T_sectionStyle" ,
            "T_sectionStyle (2)",
            "T_highlightContainer",
            "T_arrangeElementsInLine href"
        ]


    # -------------------------------------------------------
    def validate_column_count(self, df: pd.DataFrame) -> None:
        if len(df.columns) != len(self.expected_columns):
            raise ValueError(
                f"Column count mismatch. Expected {len(self.expected_columns)}, "
                f"got {len(df.columns)}."
            )

        logging.info("Column count validation passed.")

    # -------------------------------------------------------
    def validate_column_names(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.expected_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")

        logging.info("Column name validation passed.")

    # -------------------------------------------------------
    def validate_no_empty_columns(self, df: pd.DataFrame) -> None:
        empty_columns = [col for col in df.columns if df[col].dropna().empty]

        if empty_columns:
            raise ValueError(f"The following columns are completely empty: {empty_columns}")

        logging.info("Empty column validation passed.")


    # -------------------------------------------------------
    def run_all_checks(self) -> pd.DataFrame:
        df = self.dataframe
        self.validate_column_count(df)
        self.validate_column_names(df)
        self.validate_no_empty_columns(df)

        logging.info("RAW DATA VALIDATION SUCCESSFUL. Dataset is clean enough to preprocess.")
        return df
