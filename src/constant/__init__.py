import os
from os import environ
from datetime import datetime
from from_root.root import from_root

TIMESTAMP: str=datetime.now().strftime("%m_%d_Y_%H_%M_%S")

SCHEMA_FILE_PATH="config/schema.yaml"

MONGODB_URL=os.getenv("MONGODB_URL")

TARGET_COLUMN="Cost"
DB_NAME="shipement_price_db"
COLLECTION_NAME="shipment_price_data"
TEST_SIZE=0.2
ARTIFACTS_DIR=os.path.join(from_root(),"artifacts",TIMESTAMP)

# Data Ingestion
DATA_INGESTION_ARTIFACTS_DIR="DataIngestionArtifacts"
DATA_INGESTION_TRAIN_DIR="Train"
DATA_INGESTION_TEST_DIR="Test"
DATA_INGESTION_TRAIN_FILE_NAME="train.csv"
DATA_INGESTION_TEST_FILE_NAME="test.csv"

# Data Validation
DATA_VALIDATION_ARTIFACT_DIR="DataValidationArtifacts"
DATA_DRIFT_FILE_NAME="DataDriftReport.yaml"