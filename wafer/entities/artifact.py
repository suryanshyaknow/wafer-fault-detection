from dataclasses import dataclass


@dataclass
class DataValidationArtifact:
    good_data_dir: str
    archived_data_dir: str


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    training_file_path: str
    test_file_path: str


@dataclass
class DataPreparationArtifact:
    preprocessor_path: str
    clusterer_path: str
    transformed_training_file_path: str
    transformed_test_file_path: str


@dataclass
class ModelTrainingArtifact:
    cluster_based_models_dir: str


@dataclass
class ModelEvaluationArtifact:
    ...


@dataclass
class ModelPushingArtifact:
    ...
