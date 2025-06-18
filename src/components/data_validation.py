import json
from src.logger import logging
import sys
import os
import pandas as pd
from pandas import DataFrame
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from typing import Tuple,Union
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
)

class DataValidation:
    def __init__(self,
                 data_ingestion_artifacts: DataIngestionArtifacts,
                 data_validation_config:DataValidationConfig,
    ):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_validation_config=data_validation_config

    # This method is used to validate schema columns
    def validate_schema_columns(self,df:DataFrame)->bool:
        """
        
        Method Name : validate_schema_columns

        Description : This method validates schema columns of dataframe

        Output : True or False
        """
        try:
            # Checking the len of dataframe columns and schema file columns
            expected_columns=self.data_validation_config.SCHEMA_CONFIG["columns"]
            if len(df.columns)==len(expected_columns):
                validation_status=True
            else:
                logging.error(f"Schema column mismatch. Expected {len(expected_columns)} columns, got {len(df.columns)}.")
                validation_status=False
            return validation_status
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def is_numerical_column_exists(self,df:DataFrame):
        """"
        
        Method Name : is_numerical_column_exists

        Description : This method validates whether a numerical column exists in the dataframe or not

        Output : True or False
        """
        try:
            validation_status=False

            # Checking numerical schema columns with data frame numerical columns
            for column in self.data_validation_config.SCHEMA_CONFIG[
                "numerical_columns"
            ]:
                if column not in df.columns:
                    logging.info(f"Numerical column - {column} not found in dataframe")
                else:
                    validation_status=True
            return validation_status
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def is_categorical_column_exists(self,df:DataFrame)->bool:
        """
        Method Name : is_categorical_column_exists

        Description : This method validates whether a categorical column exists in the dataframe or not

        Output : True or False
        """
        try:
            validation_status=False

            # checking categorical schema columns with data frame categorical columns
            for column in self.data_validation_config.SCHEMA_CONFIG[
                "categorical_columns"]:
                if column not in df.columns:
                    logging.info(f"categorical column - {column} not found in dataframe")
                else:
                    validation_status=True
            return validation_status
            
        except Exception as e:
            raise CustomException(e,sys) from e

        
    def validate_dataset_schema_columns(self)->Tuple[bool,bool]:
        """
        
        Method Name : validate_dataset_schema_columns

        Description : This method validates schema for train dataframe and test dataframe

        Output : True or False
        """
        logging.info("Entered validate_dataset_schema_columns method of Data_Validation class")

        try:
            logging.info("Validating dataset schema columns")

            # validating schema columns for train dataframe
            train_schema_status=self.validate_schema_columns(self.train_data)
            logging.info("Validated dataset schema columns on the train set")

            # Validating schema columns for test dataframe
            test_schema_status=self.validate_schema_columns(self.test_data)
            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")
            return train_schema_status,test_schema_status
        
        except Exception as e:
            raise CustomException(e,sys) from e
            
    def validate_is_numerical_column_exists(self)->Tuple[bool,bool]:
            """
            
            Method Name : validate_is_numerical_column_exists

            Description :  This method validates whether numerical column exists for train dataframe and test dataframe or not

            Output : True or False
            """
            logging.info(
                "Entered validate_dataset_schema_for_numerical_datatype method of Data_Validation class"
            )

            try:
                logging.info("Validating dataset schema for numerical datatype")

                # Validating numerical columns with Train Dataframe
                train_num_datatype_status=self.is_numerical_column_exists(self.train_data)
                logging.info("Validated dataset schema for numerical datatype for train set")

                test_num_datatype_status=self.is_numerical_column_exists(self.test_data)
                logging.info("Validated dataset schema for numerical datatype for test set")
            
                logging.info("Exited validate_dataset_schema_for_numerical_datatype method of Data_Validation class")
                return train_num_datatype_status,test_num_datatype_status
            
            except Exception as e:
                raise CustomException(e,sys) from e
            
    def validate_is_categorical_column_exists(self)->Tuple[bool,bool]:
        """"
        
        Method Name : validate_is_categorical_column_exists

        Description : This method validates whether categorical columns exists for train dataframe and test dataframe or not

        Output : True or False
        """
        logging.info("Entered validate_dataset_schema_for_numerical_datatype method of Data_Validation class")
        
        try:
            logging.info("Validating dataset schema for numerical datatype")

            # Validating categorical columns with Train dataframe
            train_cat_datatype_status=self.is_categorical_column_exists(
                self.train_data
            )
            logging.info("Validated dataset schema for numerical datatype for train data")

            # Validating categorical columns with test dataframe
            test_cat_datatype_status=self.is_categorical_column_exists(
                self.test_data
            )
            logging.info("Validated dataset schema for numerical datatype for test data")

            logging.info("Exited validate_dataset_schema_for_numerical_datatype method of Data_validation class")

            return train_cat_datatype_status,test_cat_datatype_status
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def detect_dataset_drift(self,reference: DataFrame,production: DataFrame,get_ratio: bool = False) -> Union[bool, float]:
        """
        Detects data drift using Evidently 
        Returns overall drift bool or drift ratio depending on get_ratio.
        """
        try:
            # Define column roles/types
            data_def = DataDefinition(
                numerical_columns=self.data_validation_config.NUMERICAL_FEATURES,
                categorical_columns=self.data_validation_config.CATEGORICAL_FEATURES
            )

            # Create Evidently Datasets
            ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
            prod_ds = Dataset.from_pandas(production, data_definition=data_def)

            # Create and run Report
            report = Report(metrics=[DataDriftPreset()])
            snapshot=report.run(reference_data=ref_ds, current_data=prod_ds)
            
            # Dump the full snapshot to JSON
            report_dict = snapshot.dict()
            json_path = os.path.join(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, "data_drift_report.json")
            with open(json_path, "w") as f:
                json.dump(report_dict, f, indent=4)

            # Save report
            yaml_path = os.path.join(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, "data_drift_report.yaml")
            self.data_validation_config.UTILS.write_json_to_yaml_file(report_dict, yaml_path)
            
            # Parse drift summary from your JSON sample
            drifted = next(m for m in report_dict["metrics"] if m["metric_id"].startswith("DriftedColumnsCount"))
            drift_count = drifted["value"]["count"]
            drift_share = drifted["value"]["share"]

            # Parse per-column drift values
            column_drift = {
                m["metric_id"].split("(")[1].rstrip(")"): m["value"]
                for m in report_dict["metrics"]
                if m["metric_id"].startswith("ValueDrift(column=")
            }

            print(f"\n Drifted columns: {drift_count}, Share: {drift_share:.2f}")
            print(" Column-level drift values:")
            for col, val in column_drift.items():
                print(f"  • {col}: {val:.4f}")

            return drift_share if get_ratio else (drift_count > 0)
           

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_validation(self) -> DataValidationArtifacts:

        """
        Method Name :   initiate_data_validation

        Description :   This method initiates data validation. 
        
        Output      :   Data validation artifacts 
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")
        try:

            # Reading the Train and Test data from Data Ingestion Artifacts folder
            self.train_data = pd.read_csv(
                self.data_ingestion_artifacts.train_data_file_path
            )
            self.test_data = pd.read_csv(
                self.data_ingestion_artifacts.test_data_file_path
            )
            logging.info("Initiated data validation for the dataset")

            # Creating the Data Validation Artifacts directory
            os.makedirs(
                self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, exist_ok=True
            )
            
            logging.info(
                f"Created Artifacts directory for {os.path.basename(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR)}"
            )

            
            # Checking the dataset drift
            drift = self.detect_dataset_drift(self.train_data, self.test_data)
            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns()
            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )
            logging.info("Validated dataset schema columns")
            (
                schema_train_cat_cols_status,
                schema_test_cat_cols_status,
            ) = self.validate_is_categorical_column_exists()
            logging.info(
                f"Schema train cat cols status is {schema_train_cat_cols_status} and schema test cat cols status is {schema_test_cat_cols_status}"
            )
            logging.info("Validated dataset schema for catergorical datatype")
            (
                schema_train_num_cols_status,
                schema_test_num_cols_status,
            ) = self.validate_is_numerical_column_exists()
            logging.info(
                f"Schema train numerical cols status is {schema_train_num_cols_status} and schema test numerical cols status is {schema_test_num_cols_status}"
            )
            logging.info("Validated dataset schema for numerical datatype")

            # Checking dfist status, initially the status is None
            drift_status = None
            if (
                schema_train_cat_cols_status is True
                and schema_test_cat_cols_status is True
                and schema_train_num_cols_status is True
                and schema_test_num_cols_status is True
                and schema_train_col_status is True
                and schema_test_col_status is True
                and drift is False
            ):
                logging.info("Dataset schema validation completed")
                drift_status = True
            else:
                drift_status = False
            
            logging.info(f"Final schema + drift validation status: {drift_status}")

            # Saving data validation artifacts
            data_validation_artifacts = DataValidationArtifacts(
                data_drift_file_path=self.data_validation_config.DATA_DRIFT_FILE_PATH,
                validation_status=drift_status,
            )

            return data_validation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e