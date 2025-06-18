import shutil
import sys
from typing import Dict,Tuple,List
import dill
import xgboost
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from yaml import safe_dump
from src.constant import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def read_yaml_file(self,filename:str)->dict:
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            with open(filename,"rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def write_json_to_yaml_file(self,json_file:dict,yaml_file_path:str)->yaml:
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            data=json_file
            stream=open(yaml_file_path,"w")
            yaml.dump(data,stream)
        except Exception as e:
            raise CustomException(e,sys) from e