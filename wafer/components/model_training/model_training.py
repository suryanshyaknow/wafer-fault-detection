import os
import numpy as np
from wafer.logger import lg 
from wafer.entities.artifact import DataPreparationArtifact, ModelTrainingArtifact
from wafer.entities.config import ModelTrainingConfig
from utils.file_operations import BasicUtils
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class ModelTraining:
    """Shall be used for training the shortlisted models, finetuning them and apparently returning configurations of the built
    (and finetuned) models and their peformance measures.

    Args:    
        data_prep_artifact (DataPreparationArtifact): Takes in a `DataPreparationArtifact` object to have access to all relevant 
        configs of Data Preparation stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelTraining" class')

    data_prep_artifact: DataPreparationArtifact
    model_training_config = ModelTrainingConfig()

    def initiate(self) -> ModelTrainingArtifact:
        try:
            lg.info(f"\n{'='*27} MODEL TRAINING {'='*40}")

            ######################### Fetch the Training and Test arrays #######################################
            lg.info("fetching the transformed training and test arrays..")
            train_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_training_file_path,
                desc="Training")
            test_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_test_file_path,
                desc="Test")
            lg.info("transformed training and test arrays fetched successfully..")

            ########################### Select and Train models based on clusters ##############################
            # Configure unique clusters
            clusters = np.unique(train_arr[:, -2])

            # Traverse through clusters and find best model for each
            for i in clusters:
                ############################### Filter Cluster data ############################################
                cluster_instances = train_arr[train_arr[:, -2] == i]  # filter cluster data

                ################### Configure Features and Labels for Cluster-filtered instances ###############
                X, y = np.delete(train_arr, [-2, -1], axis=1), train_arr[:, -1]

                ################################ Training-Test Split ###########################################
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

                ######################### Fetch best Model for given Cluster ###################################
                
                
                ########################## Save best Model for given Cluster ###################################
                

            ############################## Evaluate cluster based Models on Test set ###########################


            

            ########################### Prepare the Model Training Artifact ####################################


            ...
        except Exception as e:
            lg.exception(e)
            raise e