import boto3
import pandas as pd
from src.logger.logger import logging
from io import StringIO,BytesIO
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):

        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
    
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            file_ext = os.path.splitext(file_key)[1].lower()
            file_bytes = response["Body"].read()

            if file_ext == ".csv":
                df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))

            elif file_ext == ".json":
                df = pd.read_json(StringIO(file_bytes.decode("utf-8")))

            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(BytesIO(file_bytes))

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            logging.info(
                f"Successfully loaded '{file_key}' "
                f"({file_ext}) with {len(df)} records."
            )

            return df

        except Exception as e:
            logging.exception(
                f"Failed to fetch '{file_key}' from S3 bucket '{self.bucket_name}': {e}"
            )
            return None

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
#     AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
#     BUCKET_NAME = os.environ.get("BUCKET_NAME")
#     FILE_KEY = "raw/lat_long.xlsx"  # Path inside S3 bucket

#     if not BUCKET_NAME:
#         raise RuntimeError(
#             "BUCKET_NAME environment variable not set"
#         )

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame