import os
from src.logger import logging
import sys
import pandas as pd
from typing import List,Tuple
from pandas import DataFrame
from src.constant import MODEL_CONFIG_FILE
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifacts,ModelTrainerArtifacts
)
from src.exception import CustomException

class CostModel:
    def __init__(self,preprocessing_object:object,trained_model_object:object):
        self.preprocessing_object=preprocessing_object
        self.trained_model_object=trained_model_object

    def predict(self,X)->float:
        """
        
        Method Name : predict

        Descritpion : This method predicts the data

        Output : Prediction
        """

        logging.info("Entered predict method of the class")
        try:
            # Using the trained model to get predictions
            transformed_feature=self.preprocessing_object.transform(X)
            logging.info("Used the trained model to get prediction")

            return self.trained_model_object.predict(transformed_feature)
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifacts,
                 model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config

    def get_trained_models(
            self,x_data:DataFrame,y_data:DataFrame
    )->List[Tuple[float,object,str]]:
        
        """
        
        Method Name : get_trained_model

        Description : This method lists the trained model

        Output : List of trained model
        """
        logging.info("Entered get_trained_model method of Model Trainer class")
        try:
            # Getting the model lists from model config file
            model_config=self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            models_list=list(model_config["train_model"].keys())
            logging.info("Got model list from the config file")

            # Splitting the data in x_train,y_train,x_test and y_test
            x_train,y_train,x_test,y_test=(
                x_data.drop(x_data.columns[len(x_data.columns)-1], axis=1),
                x_data.iloc[:,-1],
                y_data.drop(y_data.columns[len(y_data.columns) -1],axis=1),
                y_data.iloc[:,-1],
            )

            # Getting the trained model list
            tuned_model_list=[
                (
                    self.model_trainer_config.UTILS.get_tuned_model(
                        model_name,x_train,y_train,x_test,y_test
                    )
                )
                for model_name in models_list
            ]
            logging.info("Retrieved trained model list")
            logging.info("Exited the get_trained_models method of Model Trainer class")

            return tuned_model_list
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    # This method is used to initialize model training
    def initiate_model_trainer(self)->ModelTrainerArtifacts:
        """
        
        Method Name : initiate_model_trainer

        Description : This method initiates model training

        Output : List of trained models
        """  
        logging.info("Entered initiate_model_trainer method of Model Trainer class")
        try:
            # Creating model trainer artifacts directory
            os.makedirs(
                self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,exist_ok=True
            )
            logging.info(f"Created artifacts directory for {os.path.basename(self.model_trainer_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")

            # Loading the train array data and reading it as dataframe
            train_array=self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            train_df=pd.DataFrame(train_array)

            logging.info(f"Loaded train array from Data Transformation Artifacts directory and converted into dataframe")

            # Loading the test array data and reading it as a Dataframe
            test_array=self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            test_df=pd.DataFrame(test_array)
            logging.info(f"Loaded test array from Data Transformation artifacts directory and converted into dataframe")
            
            # Getting the models list and finding the best model with score
            list_of_trained_models=self.get_trained_models(train_df,test_df)
            logging.info("Retrieved a list of tuple of model score,model and model name")
            (
                best_model,
                best_model_score,
            )=self.model_trainer_config.UTILS.get_best_model_with_name_and_score(list_of_trained_models)

            logging.info(f"Fetched the best model : {best_model}, best_score : {best_model_score} and name")

            # Loading the pre-processor object
            preprocessor_obj_file_path=(
                self.data_transformation_artifact.transformed_object_file_path
            )
            preprocessor_obj=self.model_trainer_config.UTILS.load_object(
                preprocessor_obj_file_path
            )
            logging.info("Loaded preprocessing object")

            # Reading model config file for getting the best model score
            model_config=self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            base_model_score=float(model_config["base_model_score"])

            # Updating the model score to model config file if the model score is greater than the base model score
            if best_model_score>=base_model_score:
                # Loading cost model object with preprocessor and model
                cost_model=CostModel(preprocessor_obj,best_model)
                logging.info("Created cost model object with preprocessor and model")
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info("Created best model file path")

                # saving cost model in model artifacts directory
                model_file_path=self.model_trainer_config.UTILS.save_object(
                    trained_model_path,cost_model
                )
                logging.info("Saved the best model object path")
            else:
                logging.info("No best model found with score more than base score")
                raise CustomException("No best model found with score more than base score",sys)
            
            # saving the model trainer artifacts
            model_trainer_artifacts=ModelTrainerArtifacts(
                trained_model_file_path=model_file_path
            )

            return model_trainer_artifacts
        
        except Exception as e:
            raise CustomException(e,sys) from e