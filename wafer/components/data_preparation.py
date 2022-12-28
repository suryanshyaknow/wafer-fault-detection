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

            ################################# Fetch the Training set ###########################################
            lg.info("fetching the `wafers` training set for preparation..")
            wafers_train = pd.read_csv(
                self.data_ingestion_artifact.training_file_path)
            lg.info("..said dataset fetched successfully!")

            ################################ Drop Redundant Features ###########################################
            # fetch `Wafer` feature and features with "0 Standard Deviation" as in to drop them
            cols_to_drop = ["Wafer"]
            cols_with_zero_std = BasicUtils.get_columns_with_zero_std_dev(
                df=wafers_train, desc="feature store")
            cols_to_drop = cols_to_drop + cols_with_zero_std
            # drop fetched features form training set
            wafers_train = BasicUtils.drop_columns(
                wafers_train, cols_to_drop=cols_to_drop, desc="training")

            ########################## Separate the Features and Labels out ####################################
            X, y = BasicUtils.get_features_and_labels(
                df=wafers_train, target=[self.target], desc="training")

            ######################## Transformation: Imputation + Scaling ######################################
            # fetch the transformer and fit to the training features
            lg.info("fetching the preprocessor (Imputer + Scaler)..")
            preprocessor = DataPreparation.get_preprocessor()
            lg.info("..preprocessor fetched successfully!")
            lg.info("transforming (imputation + scaling) the feature store dataset..")
            X_trans = preprocessor.fit_transform(X)
            lg.info("..transformed the feature store dataset successfully!")

            # Saving the Preprocessor
            BasicUtils.save_object(
                file_path=self.data_prep_config.preprocessor_path,
                obj=preprocessor,
                obj_desc="preprocessor")

            ############################# Resample Training Instances ###########################################
            # fetch the SMOTE Resampler
            lg.info(
                "Resampling the data instances as our target attribute is highly imbalanced..")
            lg.info("fetching the SMOTETomek resampler..")
            resampler = DataPreparation.get_resampler()
            lg.info("..resampler fetched succesfully!")
            lg.info(
                f"Before Resampling, shape of the `training` dataset: {X_trans.shape}")
            lg.info('Resampling via SMOTETomek using sampling_strategy="auto"..')
            # Resample the feature store instances (as they are heavily imbalanced)
            X_res, y_res = resampler.fit_resample(X_trans, y)
            lg.info("resampling of `training` instances done successfully!")

            ########################### Configure the Prepared Training Array ##################################
            train_arr = np.c_[X_res, y_res]
            lg.info(
                f"After Resampling, shape of the `training` dataset: {train_arr.shape}")

            ############################ Save the Prepared Training Array ######################################
            BasicUtils.save_numpy_array(
                file_path=self.data_prep_config.prepared_training_file_path,
                arr=train_arr,
                desc="prepared training")

            ########################## Prepare the Data Preparation Artifact ###################################
            data_prep_artifact = DataPreparationArtifact(
                preprocessor_path=self.data_prep_config.preprocessor_path,
                prepared_training_file_path=self.data_prep_config.prepared_training_file_path)
            lg.info(f"Data Preparation Artifact: {data_prep_artifact}")
            lg.info("Data Preparation completed!")

            return data_prep_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
