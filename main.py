import sys

from src.signLang.logging import logger
from src.signLang.exception import CustomException

from src.signLang.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.signLang.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    logger.info('Execution has started.')
    try:
        data_ingestion = DataIngestion(database_name="signLang",
                                       collection_name="mycollection")
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_training(
            train_data_path, test_data_path))

    except Exception as e:
        raise CustomException(e, sys)
