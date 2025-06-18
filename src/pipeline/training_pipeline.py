import sys
from src.exception import CustomException
from src.logger import logging
from src.configuration.mongo_operations import MongoDBOperation
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
    )

from src.entity.config_entity import (
    DataIngestionConfig,DataValidationConfig,)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.mongodb_operation=MongoDBOperation()

        # This method is used to start data ingestion
    def start_data_ingestion(self)->DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from mongodb")
            data_ingestion=DataIngestion(
                data_ingestion_config=self.data_ingestion_config,mongo_operation=self.mongodb_operation
            )
            data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts
            
        except Exception as e:
            raise CustomException(e,sys) from e
            
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifacts)->DataValidationArtifacts:
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation=DataValidation(
                data_ingestion_artifacts=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def run_pipeline(self)->None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise CustomException(e,sys) from e