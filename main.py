import sys

from src.signLang.logging import logger
from src.signLang.exception import CustomException

from src.signLang.components.data_ingestion import DataIngestionConfig, DataIngestion


if __name__ == '__main__':
    logger.info('Execution has started.')
    try:
        data_ingestion = DataIngestion(database_name="signLang",
                                       collection_name="mycollection")
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    except Exception as e:
        raise CustomException(e, sys)
