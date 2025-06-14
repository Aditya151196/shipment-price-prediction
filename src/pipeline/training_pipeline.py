import sys
from src.exception import CustomException
from src.logger import logging
from src.configuration.mongo_operations import MongoDBOperation
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    )

from src.entity.config_entity import (
    DataIngestionConfig,
    )

from src.components.data_ingestion import DataIngestion

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

        # This method is used to start data ingestion
        def start_data_ingestion(self)->DataIngestionArtifacts:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            try:
                logging.info("Getting the data from mongodb")
                data_ingestion=DataIngestion(
                    data_ingestion_config=self.data_ingestion_config,mongo_operation=self.mongo
                )
                data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
                logging.info("Got the train_set and test_set from mongodb")
                logging.info("Exited the start_data_ingestion method of TrainPipeline class")
                return data_ingestion_artifacts
            
            except Exception as e:
                raise CustomException(e,sys) from e
            
        
        def run_pipeline(self)->None:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            try:
                data_ingestion_artifact=self.start_data_ingestion()

            except Exception as e:
                raise CustomException(e,sys) from e