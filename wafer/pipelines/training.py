from wafer.logger import lg
from dataclasses import dataclass
import os
import argparse
from wafer.components.data_validation import DataValidation
from wafer.components.data_ingestion import DataIngestion


@dataclass
class TrainingPipeline:
    lg.info("Training Pipeline begins now..")
    lg.info(f"Entered the {os.path.basename(__file__)[:-3]}.TrainingPipeline")

    new_data: bool = False

    def begin(self):
        try:
            ######################### DATA VALIDATION ######################################
            data_validation = DataValidation()
            validation_artifact = data_validation.initiate()

            ######################### DATA INGESTION #######################################
            data_ingestion = DataIngestion(
                data_validation_artifact=validation_artifact, new_data=self.new_data)
            ingestion_artifact = data_ingestion.initiate()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data", default=False)
    parsed_args = parser.parse_args()
    training_pipeline = TrainingPipeline(new_data=parsed_args.new_data)
    training_pipeline.begin()
