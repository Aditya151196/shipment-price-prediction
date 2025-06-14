import os
from os import environ
from datetime import datetime
from from_root.root import from_root

TIMESTAMP: str=datetime.now().strftime("%m_%d_Y_%H_%M_%S")

TARGET_COLUMN="Cost"
DB_NAME="shipment-price-db"
COLLECTION_NAME="shipment-price-data"
TEST_SIZE=0.2
ARTIFACTS_DIR=os.path.join(from_root(),"artifacts",TIMESTAMP)

# Data Ingestion
DATA_INGESTION_ARTIFACTS_DIR="DataIngestionArtifacts"
DATA_INGESTION_TRAIN_DIR="Train"
DATA_INGESTION_TEST_DIR="Test"
DATA_INGESTION_TRAIN_FILE_NAME="train.csv"
DATA_INGESTION_TEST_FILE_NAME="test.csv"