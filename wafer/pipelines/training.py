from wafer.logger import lg
from dataclasses import dataclass
import os
from wafer.components.data_validation import DataValidation


@dataclass
class TrainingPipeline:
    lg.info("Training Pipeline begins now..")
    lg.info(f"Entered the {os.path.basename(__file__)[:-3]}.TrainingPipeline")

    def begin(self):
        try:
            ######################### DATA VALIDATION ######################################
            data_validation = DataValidation()
            validation_artifact = data_validation.initiate()

            ######################### DATA INGESTION #######################################


            ######################### DATA TRANSFORMATION ##################################
            

            ######################### MODEL TRAINING #######################################
            

            ######################### MODEL EVALUATION #####################################
            

            ######################### MODEL PUSHING ########################################
            
            ...
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("Training Pipeline ran with success!")


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.begin()
