import os
import sys
import pymongo
import pandas as pd
from pathlib import Path

from src.signLang.logging import logger
from src.signLang.exception import CustomException
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class DataIngestionConfig:
    raw_data_file_path: str = os.path.join('data', 'raw.csv')
    train_data_file_path: str = os.path.join('data', 'train.csv')
    test_data_file_path: str = os.path.join('data', 'test.csv')


class DataIngestion:
    def __init__(self, database_name, collection_name):
        self.data_ingestion_config = DataIngestionConfig()
        self.database_name = database_name
        self.collection_name = collection_name

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion has begun.")
            if not os.path.exists('data/raw.csv'):
                URL = os.getenv('URL')

                logger.info("Establishing connection to the database.")
                client = pymongo.MongoClient(URL)
                db = client[self.database_name]
                collection = db[self.collection_name]

                logger.info(
                    "Fetch was successful! Converting into dataframe...")
                cursor = collection.find({})
                df = pd.DataFrame(list(cursor))

                client.close()
            else:
                df = pd.read_csv('data/raw.csv')

            os.makedirs(os.path.dirname(
                self.data_ingestion_config.raw_data_file_path), exist_ok=True)

            logger.info(
                "Mongodb connection is closed. Initiating train & test split...")
            train_df = df[:27455]
            test_df = df[27455:]

            print(f"train data size: {train_df.shape}")
            print(f"test data size: {test_df.shape}")

            df.to_csv(self.data_ingestion_config.raw_data_file_path,
                      index=False, header=True)
            train_df.to_csv(self.data_ingestion_config.train_data_file_path,
                            index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_file_path,
                           index=False, header=True)
            logger.info("Data Ingestion was successfull.")
            return (
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
