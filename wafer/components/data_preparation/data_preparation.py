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
    """Shall be used for preparing the data, before feeding into ML algorithms for training."""
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
            e: Raises exception should any pops up while preparing the data.

        Returns:
            DataPreparationArtifact: Contains configurations of all the relevant artifacts that shall be made while
            preparing the data.
        """
        try:
            lg.info(f"\n{'='*27} DATA PREPARATION {'='*40}")

            ########################### Fetch the `Feature Store` dataset #######################################
            lg.info("fetching the feature store dataset for preparation..")
            wafers = pd.read_csv(
                self.data_ingestion_artifact.feature_store_file_path)
            lg.info("..said dataset fetched successfully!")

            ################################ Drop Redundant Features ###########################################
            # fetch `Wafer` feature and features with "0 Standard Deviation" as in to drop them
            cols_to_drop = ["Wafer"]
            cols_with_zero_std = BasicUtils.get_columns_with_zero_std_dev(
                df=wafers, desc="feature store")
            cols_to_drop = cols_to_drop + cols_with_zero_std
            # drop fetched features form training set
            wafers = BasicUtils.drop_columns(
                wafers, cols_to_drop=cols_to_drop, desc="feature store")

            ########################## Separate the Features and Labels out ####################################
            X, y = BasicUtils.get_features_and_labels(
                df=wafers, target=[self.target], desc="feature store")

            ######################## Transformation: Imputation + Scaling ######################################
            # fetch the transformer and fit to the training features
            lg.info("fetching the preprocessor (Imputer + Scaler)..")
            preprocessor = DataPreparation.get_preprocessor()
            lg.info("..preprocessor fetched successfully!")
            lg.info("transforming (imputation + scaling) the feature store dataset..")
            X_transformed = preprocessor.fit_transform(X)
            lg.info("..transformed the feature store dataset successfully!")

            # Saving the Preprocessor
            BasicUtils.save_object(
                file_path=self.data_prep_config.preprocessor_path,
                obj=preprocessor,
                obj_desc="preprocessor")

            ################################# Cluster Data Instances #############################################
            cluster_train_data = ClusterDataInstances(
                X=X_transformed, desc="training", data_prep_config=self.data_prep_config)
            clusterer, X_clus = cluster_train_data.create_clusters()

            # Save the Clusterer
            lg.info("saving the Clusterer..")
            BasicUtils.save_object(
                file_path=self.data_prep_config.clusterer_path, obj=clusterer, obj_desc="KMeans Clusterer")
            lg.info("..said Clusterer saved successfully!")

            ################################ Resample Data Instances ############################################
            # fetch the SMOTE Resampler
            lg.info(
                "Resampling the data instances as our target attribute is highly imbalanced..")
            lg.info("fetching the SMOTETomek resampler..")
            resampler = DataPreparation.get_resampler()
            lg.info("..resampler fetched succesfully!")
            lg.info(
                f"Before Resampling, shape of the `feature store` dataset: {X_clus.shape}")
            lg.info('Resampling via SMOTETomek using sampling_strategy="auto"..')
            # Resample the feature store instances (as they are heavily imbalanced)
            X_res, y_res = resampler.fit_resample(X_clus, y)
            lg.info("resampling of `feature store` instances done successfully!")

            ############################# Configure Feature Store array ########################################
            wafers_arr = np.c_[X_res, y_res]
            lg.info(
                f"After Resampling, shape of the `feature store` dataset: {wafers_arr.shape}")

            ############################ Save the Feature Store array ##########################################
            BasicUtils.save_numpy_array(
                file_path=self.data_prep_config.transformed_feature_store_file_path,
                arr=wafers_arr,
                desc="(transformed) feature store"
            )

            ########################## Prepare the Data Preparation Artifact ###################################
            data_prep_artifact = DataPreparationArtifact(
                preprocessor_path=self.data_prep_config.preprocessor_path,
                clusterer_path=self.data_prep_config.clusterer_path,
                transformed_feature_store_file_path=self.data_prep_config.transformed_feature_store_file_path
            )
            lg.info(f"Data Preparation Artifact: {data_prep_artifact}")
            lg.info("Data Preparation completed!")

            return data_prep_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
