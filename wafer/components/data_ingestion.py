import os
import pandas as pd
import json
from wafer.logger import lg
from wafer.CONFIG import DatabaseConfig
from wafer.entities.config import DataIngestionConfig
from wafer.entities.artifact import DataIngestionArtifact, DataValidationArtifact
from wafer.utils.db_operations import MongoDBOperations
from dataclasses import dataclass


@dataclass
class DataIngestion:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataIngestion" class')

    data_validation_artifact: DataValidationArtifact
    new_data: bool = False
    database_config = DatabaseConfig()
    data_ingestion_config = DataIngestionConfig()

    def initiate(self) -> DataIngestionArtifact:
        try:
            lg.info(f"\n{'='*27} DATA INGESTION {'='*40}")

            ######################## Setup MongoDB Credentials and Operations ##################################
            lg.info("setting up MongoDB credentials and operations..")
            db_ops = MongoDBOperations(
                connection_string=self.database_config.mongodb_url,
                database_name=self.database_config.database_name,
                collection_name=self.database_config.collection_name)
            lg.info(f"{db_ops}: setup done with success!")

            if self.new_data:  # dump data to MongoDB only if there's new data

                ####################### Fetch GoodRawData and Dump into MongoDB ####################################
                lg.info(
                    "fetching data from `GoodRawData` dir and dumping it into the database..")
                good_data_dir = self.data_validation_artifact.good_data_dir

                for csv_file in os.listdir(good_data_dir):
                    df = pd.read_csv(os.path.join(good_data_dir, csv_file), na_values="null")  # read dataframe with na_values as "null"
                    df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)  # rename the unnamed column to "Wafer"
                    # convert the df into "dumpable into MongoDB" format --json
                    all_records = list(json.loads(df.T.to_json()).values())
                    lg.info(f"\"{csv_file}\" data fetched successfully!")

                    db_ops.dumpData(records=all_records,
                                    data_desc="validated Training data")
                    lg.info(
                        f"\"{csv_file}\"'s data dumped into the database with success!")
                lg.info(
                    f"successfully dumped all data from `GoodRawData` dir into database {self.database_config.database_name}")

            ################## Fetch data from Database and Prepare `Feature Store file` #######################
            lg.info("readying the `feature store file`..")
            feature_store_file = db_ops.getDataAsDataFrame()
            lg.info(f"Shape of `feature store file`: {feature_store_file.shape}")
            lg.info("saving the `feature store file`..")
            # make sure the dir for saving `Feature Store file` does exist
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            feature_store_file.to_csv(
                self.data_ingestion_config.feature_store_file_path, index=None)
            lg.info("..prepared `feature store file` successfully!")

            ########################### Prepare the Data Ingestion Artifact ####################################
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path)
            lg.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            lg.info("DATA INGESTION completed!")

            return data_ingestion_artifact
            ...
        except Exception as e:
            lg.exception(e)
