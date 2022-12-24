import pandas as pd
import numpy as np
import os
from wafer.logger import lg
from wafer.utils.file_operations import BasicUtils
from wafer.entities.config import DataPreparationConfig, BaseConfig
from wafer.entities.artifact import DataIngestionArtifact, DataPreparationArtifact
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from wafer.components.data_preparation.clustering import ClusterDataInstances
from imblearn.combine import SMOTETomek
from dataclasses import dataclass


@dataclass
class DataPreparation:
    """Shall be used for preparing the training and test data before feeding them into ML algorithms."""
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataTransformation" class')

    data_ingestion_artifact: DataIngestionArtifact
    data_prep_config = DataPreparationConfig()
    target = BaseConfig().target

    @classmethod
    def get_preprocessor(cls) -> Pipeline:
        """Returns a `Custom Pipeline` for numerical attributes of the said dataset. Pipeline contains 
        `KNNImputer` and `RobustScaler` to transform the features of the very same dataset.

        Raises:
            e: Throws exception if any error pops up while building or returning the preprocessor.

        Returns:
            Pipeline: Custom Pipeline for the numerical features of the said dataset. 
        """
        try:
            ########################## Pipeline for Numerical Atts ############################################
            preprocessing_pipeline = Pipeline(
                steps=[('KNN IMputer', KNNImputer(n_neighbors=3)),
                       ('Robust Scaler', RobustScaler())])

            return preprocessing_pipeline
            ...
        except Exception as e:
            lg.info(e)
            raise e

    @classmethod
    def get_resampler(cls, sampling_strategy: str = "auto") -> SMOTETomek:
        """Returns the SMOTETomek resampling object for resampling data instances if data's imbalanced. 

        Args:
            sampling_strategy (str, optional): Sampling strategy for resampling the data instances at hand. Defaults to "auto".

        Raises:
            e: Throws exception if any error pops up while building or returning SMOTETomek resampling object.

        Returns:
            SMOTETomek: SMOTETomek resampling object.
        """
        try:
            smote_resampler = SMOTETomek(
                sampling_strategy=sampling_strategy)
            return smote_resampler
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def initiate(self) -> DataPreparationArtifact:
        """Initiates the Data Preparation stage of the training pipeline.

        Raises:
            e: Raises exception ahould any pops up while preparing data.

        Returns:
            DataPreparationArtifact: Contains configurations of all the relevant artifacts that shall be made while
            preparing the data.
        """
        try:
            lg.info(f"\n{'='*27} DATA PREPARATION {'='*40}")

            ########################### Fetch the Training and Test datasets ###################################
            lg.info("fetching the training and test sets for transformation..")
            training_set = pd.read_csv(
                self.data_ingestion_artifact.training_file_path)
            test_set = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            lg.info("training and test sets fetched successfully!")

            ################################ Drop Redundant Features ###########################################
            # fetch `Wafer` feature and features with "0 Standard Deviation" as in to drop them
            cols_to_drop = ["Wafer"]
            cols_with_zero_std = BasicUtils.get_columns_with_zero_std_dev(
                df=training_set, desc="training")
            cols_to_drop = cols_to_drop + cols_with_zero_std
            # drop fetched features form training set
            training_set = BasicUtils.drop_columns(
                training_set, cols_to_drop=cols_to_drop, desc="training")
            # drop fetched features form test set
            test_set = BasicUtils.drop_columns(
                test_set, cols_to_drop=cols_to_drop, desc="test")

            ################ Separate the Features and Labels out from the Training and Test sets ##############
            X_train, y_train = BasicUtils.get_features_and_labels(
                df=training_set, target=[self.target], desc="training")
            X_test, y_test = BasicUtils.get_features_and_labels(
                df=test_set, target=[self.target], desc="test")

            ######################## Transformation: Imputation + Scaling ######################################
            # fetch the transformer and fit to the training features
            lg.info("fetching the preprocessor (Imputer + Scaler)..")
            preprocessor = DataPreparation.get_preprocessor()
            lg.info("..preprocessor fetched successfully!")
            X_train_transformed = preprocessor.fit_transform(X_train)
            lg.info("..training set transformed successfully!")

            # transform the test features
            lg.info("transforming the test set accordingly..")
            X_test_transformed = preprocessor.transform(X_test)
            lg.info("..test set transformed successfully!")

            # Saving the Preprocessor
            BasicUtils.save_object(
                file_path=self.data_prep_config.preprocessor_path,
                obj=preprocessor,
                obj_desc="preprocessor")

            ########################### Clustering of Data Instances ############################################
            cluster_train_data = ClusterDataInstances(
                X=X_train_transformed, desc="training", data_prep_config=self.data_prep_config)
            clusterer, X_train_clus = cluster_train_data.create_clusters()

            # Cluster Test instances
            lg.info("Clustering the test instances..")
            y_kmeans_test = clusterer.predict(X_test_transformed)
            X_test_clus = np.c_[X_test_transformed, y_kmeans_test]
            lg.info("..got test instances clustererd successfully!")

            # Save the Clusterer
            lg.info("saving the Clusterer..")
            BasicUtils.save_object(
                file_path=self.data_prep_config.clusterer_path, obj=clusterer, obj_desc="KMeans Clusterer")
            lg.info("..said Clusterer saved successfully!")

            ########################### Resampling of Training Data Instances ##################################
            # fetch the SMOTE Resampler
            lg.info(
                "Resampling the data instances as our target attribute is highly imbalanced..")
            lg.info("fetching the SMOTETomek resampler..")
            resampler = DataPreparation.get_resampler()
            lg.info("..resampler fetched succesfully!")
            lg.info(
                f"Before Resampling, shape of the `training set`: {training_set.shape}")
            lg.info('Resampling via SMOTETomek using sampling_strategy="auto"..')
            # Resample the training instances (as they are heavily imbalanced)
            X_train_res, y_train_res = resampler.fit_resample(
                X_train_clus, y_train)
            lg.info("resampling of training instances done successfully!")

            ######################### Configure Training and Test arrays #######################################
            # Training Array
            training_arr = np.c_[X_train_res, y_train_res]
            lg.info(
                f"After Resampling, shape of the `training set`: {training_arr.shape}")
            # Test Array
            test_arr = np.c_[X_test_clus, y_test]

            ########################### Save Training and Test arrays ##########################################
            # Saving the Training Array
            BasicUtils.save_numpy_array(
                file_path=self.data_prep_config.transformed_training_file_path,
                arr=training_arr,
                desc="training"
            )
            # Saving the Test Array
            BasicUtils.save_numpy_array(
                file_path=self.data_prep_config.transformed_test_file_path,
                arr=test_arr,
                desc="test"
            )

            ########################## Prepare the Data Preparation Artifact ###################################
            data_prep_artifact = DataPreparationArtifact(
                preprocessor_path=self.data_prep_config.preprocessor_path,
                clusterer_path=self.data_prep_config.clusterer_path,
                transformed_training_file_path=self.data_prep_config.transformed_training_file_path,
                transformed_test_file_path=self.data_prep_config.transformed_test_file_path
            )
            lg.info(f"Data Preparation Artifact: {data_prep_artifact}")
            lg.info("Data Preparation completed!")

            return data_prep_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
