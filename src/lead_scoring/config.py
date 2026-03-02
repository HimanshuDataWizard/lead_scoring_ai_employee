from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "lead_scoring_logistics_Dataset.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ID_COLUMN = "lead_code"
TARGET_COLUMN = "converted"
NUMERIC_FEATURES = [
    "company_size",
    "prev_purchases",
    "response_time",
    "last_contact",
    "deal_value",
]
CATEGORICAL_FEATURES = ["industry", "region", "source"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
TEST_SIZE = 0.2
