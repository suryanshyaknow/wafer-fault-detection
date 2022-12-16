import os
from wafer.logger import lg
from datetime import datetime
from dataclasses import dataclass


RAW_DATA_DIR = "training_batch_files"
TRAINING_SCHEMA = "schema_training.json"
PREDICTION_SCHEMA = "schema_prediction.json"


@dataclass
class BaseConfig:
    project: str = "wafer-fault-detection"
    target: str = "Good/Bad"


@dataclass
class DataSourceConfig:
    raw_data_dir = os.path.join(os.getcwd(), RAW_DATA_DIR)
    training_schema = os.path.join(os.getcwd(), TRAINING_SCHEMA)
    prediciton_schema = os.path.join(os.getcwd(), PREDICTION_SCHEMA)


@dataclass
class TrainingArtifactsConfig:
    try:
        artifacts_dir: str = os.path.join(
            os.getcwd(), "artifacts", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        ...
    except Exception as e:
        lg.exception(e)


class DataValidationConfig:
    def __init__(self):
        try:
            training_artifacts_config = TrainingArtifactsConfig()

            self.data_validation_dir = os.path.join(training_artifacts_config.artifacts_dir, "data_validation")
            
            self.good_data_dir = os.path.join(self.data_validation_dir, "good_raw_data")
            self.bad_data_dir = os.path.join(self.data_validation_dir, "bad_raw_data")
            self.archived_data_dir = os.path.join(self.data_validation_dir, "archived_data")
            ...
        except Exception as e:
            lg.exception(e)
