import sys
import os
from src.logger import logging
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.exception import CustomException
from src.configuration.mongo_operations import MongoDBOperation
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.constant import TEST_SIZE

class DataIngestion:
    def __init__(self,
                 data_ingestion_config:DataIngestionConfig,
                 mongo_operation:MongoDBOperation):
        self.data_ingestion_config=data_ingestion_config
        self.mongo_operation=mongo_operation

    def get_data_from_mongodb(self)->DataFrame:
        logging.info("Entered get_data_from_mongodb method of Data_Ingestion class")
        try:
            logging.info("Getting the dataframe from mongodb")

            df=self.mongo_operation.get_collection_as_dataframe(
                self.data_ingestion_config.DB_NAME,
                self.data_ingestion_config.COLLECTION_NAME
            )

            logging.info("Got the dataframe from mongodb")
            logging.info(df.shape)
            logging.info("Exited the get_data_from_mongodb method of data_ingestion class")
            

            return df
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def split_data_as_train_test(self,df:DataFrame)->Tuple[DataFrame,DataFrame]:
        logging.info("Entered split_data_as_train_test method of Data ingestion class")
        try:
            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,exist_ok=True
            )
            train_data,test_data=train_test_split(df,test_size=TEST_SIZE,train_size=0.8)
            logging.info("Performed train test split on the dataframe")

            os.makedirs(
                self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR,exist_ok=True
            )

            logging.info(
                f"Created  {os.path.basename(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR)} directory"
            )

            os.makedirs(
                self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR,exist_ok=True
            )
            logging.info(
                f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory"
            )

            train_data.to_csv(
                self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                index=False,
                header=True
            )

            test_data.to_csv(
                self.data_ingestion_config.TEST_DATA_FILE_PATH,
                index=False,
                header=True
            )

            logging.info("Converted Train dataframe and test dataframe into csv")
            logging.info(
                f"Saved {os.path.basename(self.data_ingestion_config.TRAIN_DATA_FILE_PATH)}, \
                    {os.path.basename(self.data_ingestion_config.TEST_DATA_FILE_PATH)} in \
                    {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)}"
            )

            logging.info(
                "Exited split_data_as_train_test method of Data Ingestion class"
            )
            return train_data,test_data
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_data_ingestion(self)->DataIngestionArtifacts:
        logging.info("Entered initiate data ingestion method of data_ingestion class")

        try:
            df=self.get_data_from_mongodb()
            #df1=df.drop(self.data_ingestion_config.DROP_COLS,axis=1)
            df=df.dropna()
            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(df)

            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifacts=DataIngestionArtifacts(
                train_data_file_path=self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                test_data_file_path=self.data_ingestion_config.TEST_DATA_FILE_PATH
            )

            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e,sys) from e