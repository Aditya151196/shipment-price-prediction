import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils.main_utils import MainUtils
import warnings

warnings.filterwarnings("ignore")

# === Change this if your path is different ===
TRAIN_ARRAY_PATH = os.path.join("artifacts", "DataTransformationArtifacts", "train.npy")
TEST_ARRAY_PATH = os.path.join("artifacts", "DataTransformationArtifacts", "test.npy")

def main():
    print("===== Loading transformed train and test arrays =====")
    utils = MainUtils()

    if not os.path.exists(TRAIN_ARRAY_PATH) or not os.path.exists(TEST_ARRAY_PATH):
        raise FileNotFoundError("Train or test numpy array file not found at specified path")

    train_array = utils.load_numpy_array_data(TRAIN_ARRAY_PATH)
    test_array = utils.load_numpy_array_data(TEST_ARRAY_PATH)

    train_df = pd.DataFrame(train_array)
    test_df = pd.DataFrame(test_array)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print("Sample Train DataFrame:\n", train_df.head())
    print("Target column distribution:\n", train_df.iloc[:, -1].describe())

    # Splitting features and labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    print("\n===== Training XGBRegressor manually =====")
    model = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=5, colsample_bytree=0.3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    print("\n===== Model Evaluation =====")
    print(f"Manual R² Score on Test Data: {score:.4f}")

if __name__ == "__main__":
    main()
