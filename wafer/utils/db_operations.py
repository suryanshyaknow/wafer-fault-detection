import pymongo
import pandas as pd
from wafer.logger import lg
import os
from typing import List
from dataclasses import dataclass


@dataclass
class MongoDBOperations:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.dBOperations" class')

    connection_string: str
    database_name: str
    collection_name: str
    client = None
    database = None
    collection = None

    def establishConnectionToMongoDB(self):
        """This method establishes the connection to the desired MongoDB Cluster.
        """
        try:
            lg.info("establishing the connection to MongoDB..")
            self.client = pymongo.MongoClient(self.connection_string)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("connection established successfully!")

    def selectDB(self):
        """This method chooses the desired dB from the MongoDB Cluster.
        """
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.database_name]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'"{self.database_name}" database chosen succesfully!')

    def createOrselectCollection(self):
        """This method shall create the desired collection in the selected database of the MongoDB Cluster.
        """
        try:
            self.selectDB()

            collections = self.database.list_collection_names()
            lg.info(
                f"looking for the collection \"{self.collection_name}\" in the database..")
            if self.collection_name in collections:
                lg.info(
                    f"collection found! selecting the collection {self.collection}..")
                self.collection = self.database[self.collection_name]
                lg.info("..said collection selected successfully")
            else:
                lg.warning(
                    f"collection \"{self.collection_name}\" not found in the database, gotta create it..")
                lg.info(
                    f"creating the collection \"{self.collection_name}\"..")
                self.collection = self.database[self.collection_name]
                lg.info("..said collection created successfully!")
        except Exception as e:
            lg.exception(e)

    def dumpData(self, records: List, data_desc: str):
        """Dumps the desired bulk data (to be parameterized in a form of list) to the

        Args:
            records (List): The bulk data that's to be dumped into the collection (in a form of List).
            data_desc (str): Description of the data that's to be dumped.
        """
        try:
            self.createOrselectCollection()

            lg.info(
                f"dumping the \"{data_desc}\" to the collection \"{self.collection_name}\"..")
            self.collection.insert_many(records)
            ...
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(f"dumped {data_desc} with success!")

    def getDataAsDataFrame(self) -> pd.DataFrame:
        """This method prepares a feature-store-file out of all the data from the selecetd database.

        Returns:
            pandas.DataFrame: Feature-store-file as pandas dataframe.
        """
        try:
            self.createOrselectCollection()
            lg.info(
                f'fetching all the data from collection "{self.collection_name}" of database "{self.database_name}"..')
            df = pd.DataFrame(list(self.collection.find()))
            lg.info("data readied as the dataframe!")
            df.drop(columns=["_id"], inplace=True)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("returning the database..")
            return df


if __name__ == "__main__":
    MongoDBOperations()
