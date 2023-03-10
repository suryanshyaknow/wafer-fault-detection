import os
import json
import joblib
from wafer.logger import lg
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class BasicUtils:
    """Shall be used for accessing basic utilities methods."""

    @classmethod
    def read_json_file(cls, file_path: str, file_desc: str) -> Dict:
        """Loads and returns the json file content loacted at `file_path`, if there's one and throws exception 
        if there's none.

        Args:
            file_path (str): Location of the json file that's to be loaded.
            file_desc (str): Description of said json file.

        Raises:
            e: Throws relevant exception if any error pops while loading or returning the json file's content.

        Returns:
            Dict: Json file's content.
        """
        try:
            lg.info(
                f"fetching the data from the \"{file_desc}\" lying at \"{file_path}\"..")
            with open(file_path, 'r') as f:
                data = json.load(f)
                lg.info("data fetched successfully!")
            f.close()

            return data
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def get_features_and_labels(cls, df: pd.DataFrame, target: List, desc: str) -> Tuple:
        """Returns the desired features and labels as pandas Dataframe in regard to the said target
        column name.

        Args:
            df (pd.DataFrame): Dataframe whose features and labels are to be returned.
            target (List): List of target column names to be included in the labels dataframe.
            desc (str): Description of the said dataframe.

        Raises:
            e: Throws relevant exception if any error pops while separating features and labels out.

        Returns:
            Tuple (pd.DataFrame, pd.DataFrame): Tuple of features pandas dataframe and labels pandas 
            dataframe respectively.
        """
        try:
            lg.info(
                f'fetching the input features and target labels out from the "{desc}" dataframe..')
            features = df.drop(columns=target)
            labels = df[target]

            lg.info("returning the said input features and dependent labels..")
            return features, labels
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def drop_columns(cls, df: pd.DataFrame, cols_to_drop: List, desc: str) -> pd.DataFrame:
        """Drops the desired columns from the provided dataframe and returns the consequent dataset.

        Args:
            df (pd.DataFrame): Dataset whose columns have to be dropped.
            cols_to_drop (List): List of exact existing column names which are to be dropped.
            desc (str): Description of the provided dataset.

        Raises:
            e: Throws relevant exception if any error pops while drpping columns.

        Returns:
            pd.DataFrame: Consequent dataframe after its desired columns have been dropped.
        """
        try:
            lg.info(f"\nColumns to be dropped: \n{cols_to_drop}")
            lg.info(f"dropping above columns from the \"{desc}\" dataset..")
            df_new = df.drop(columns=cols_to_drop, axis=1)
            lg.info("..said columns dropped successfully!")
            return df_new
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def get_columns_with_certain_missing_thresh(cls, df: pd.DataFrame, desc: str, missing_thresh=.7) -> List:
        """Returns columns names having missing values more than or equal to a certain `missing_thresh`.

        Args:
            df (pd.DataFrame): Dataset from which said columns names are to be fetched and returned.
            missing_thresh (float, optional): Threshold for missing values. Defaults to .7.
            desc (str): Description of the said dataframe.

        Raises:
            e: Throws relevant exception should any kinda error pops up while fetching and returing 
            said columns names.

        Returns:
            List: List of columns names having missing ratio more than `missing_thresh`.
        """
        try:
            cols_missing_ratios = df.isna().sum().div(df.shape[0])
            lg.info(
                f'fetching column names having missing values more than a {missing_thresh*100}% from "{desc}" dataset..')
            cols_to_drop = list(
                cols_missing_ratios[cols_missing_ratios > missing_thresh].index)
            lg.info("..said columns fetched successfully!")
            lg.info(f"\n..Columns with missing ratio more than {missing_thresh}: \n{cols_to_drop}")
            return cols_to_drop
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def get_columns_with_zero_std_dev(cls, df: pd.DataFrame, desc: str) -> List:
        """Returns a list of column names that have zero standard deviation, from the provided dataframe.

        Args:
            df (pd.DataFrame): Dataset from which said column names gotta be fetched.
            desc (str): Description of the said dataframe.

        Raises:
            e: Throws relevant exception if any error pops while fetching the columns with `zero standard deviation`.

        Returns:
            List: List of column names having zero standard deviation, of the given dataset.
        """
        try:
            cols_with_zero_std_dev = []
            # first and foremost, fetch only the numerical columns
            num_cols = [col for col in df.columns if df[col].dtype != 'O']
            lg.info(
                f"fetching column names having `zero standard deviation` from the \"{desc}\" dataset..")
            for col in num_cols:
                if df[col].std(skipna=True) == 0:
                    cols_with_zero_std_dev.append(col)
            lg.info(f"..said column names fetched successfully!")
            lg.info(f"\n..Columns with 0 Standard Deviation: \n{cols_with_zero_std_dev}")

            return cols_with_zero_std_dev
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def save_object(cls, file_path: str, obj: object, obj_desc: str) -> None:
        """Saves the desired object at the said desired location.

        Args:
            file_path (str): Location where the object is to be stored.
            obj (object): Object that is to be stored.
            obj_desc (str): Object's description.

        Raises:
            e: Throws relevant exception if any error pops while saving the desired object.
        """
        try:
            lg.info(f'Saving the "{obj_desc}" at "{file_path}"..')
            obj_dir = os.path.dirname(file_path)
            os.makedirs(obj_dir, exist_ok=True)
            joblib.dump(obj, open(file_path, 'wb'))
            lg.info(f'"{obj_desc}" saved successfully!')
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def load_object(cls, file_path: str, obj_desc: str) -> object:
        """Loads and returns the desired object located at `file_path`.

        Args:
            file_path (str): Location path of the object that's to be loaded.
            obj_desc (str): Description of the said object.

        Raises:
            e: Throws relevant exception if any error pops up while loading or returning the desired object.

        Returns:
            object: Loaded object.
        """
        try:
            lg.info(f'loading the "{obj_desc}"..')
            if not os.path.exists(file_path):
                lg.error(
                    'Uh oh! Looks like the given file path or the object doesn\'t even exist!')
                raise Exception(
                    'Uh oh! Looks like the given file path or the object doesn\'t even exist!')

            lg.info(f'"{obj_desc}" loaded successfully!')
            return joblib.load(open(file_path, 'rb'))
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def save_dataframe_as_csv(cls, file_path: str, df: pd.DataFrame, desc: str):
        """Saves the given dataframe as `csv` file at the desired `file_path` location.

        Args:
            file_path (str): Location where the dataframe is to be stored.
            df (pd.DataFrame): Dataframe which is to be saved.
            desc (str): Description of the given dataframe.

        Raises:
            e: Throws relevant exception if any error pops up while saving the given dataframe as csv file.
        """
        try:
            lg.info(f'Saving the "{desc} dataset" at "{file_path}"..')
            # Making sure the dir to store the given dataframe does exist
            dir = os.path.dirname(file_path)
            os.makedirs(dir, exist_ok=True)

            df.to_csv(path_or_buf=file_path, index=None)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(f'"{desc} dataset" saved successfully at "{file_path}"!')

    @classmethod
    def load_csv_as_dataframe(cls, file_path: str, desc: str):
        """Loads the desired csv file as pandas dataframe from the given `file_path` location.

        Args:
            file_path (str): Location from where the csv file is to be loaded.
            desc (str): Description of the csv file.

        Raises:
            e: Throws relevant exception if any error pops up while loading or returning the desired csv file.
        """
        try:
            lg.info(f'Loading the "{desc} array" from "{file_path}"..')

            if not os.path.exists(file_path):
                lg.error(
                    'Uh Oh! Looks like the given file path doesn\'t even exist!')
                raise Exception(
                    'Uh Oh! Looks like the given file path doesn\'t even exist!')

            lg.info(f'"{desc} dataset" loaded successfully!')
            return pd.read_csv(file_path)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def save_numpy_array(cls, file_path: str, arr: np.array, desc: str):
        """Saves the numpy array at the desired `file_path` location.

        Args:
            file_path (str): Location where the numpy array is to be stored.
            arr (np.array): Numpy array which is to be stored.
            desc (str): Description of the numpy array.

        Raises:
            e: Throws relevant exception if any error pops up while saving the given numpy array.
        """
        try:
            lg.info(f'Saving the "{desc} array" at "{file_path}"..')
            # Making sure the dir do exist
            dir = os.path.dirname(file_path)
            os.makedirs(dir, exist_ok=True)
            with open(file_path, "wb") as f:
                np.save(f, arr)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(f'"{desc} array" saved successfully!')

    @classmethod
    def load_numpy_array(cls, file_path: str, desc: str):
        """Loads the desired numpy array from the desired `file_path` location.

        Args:
            file_path (str): Location from where the numpy array is to be fetched.
            desc (str): Description of the numpy array.

        Raises:
            e: Throws relevant exception if any error pops up while loading or returning the desired numpy array.
        """
        try:
            lg.info(f'Loading the "{desc} array" from "{file_path}"..')

            if not os.path.exists(file_path):
                lg.error(
                    'Uh Oh! Looks like the said file path or the numpy array doesn\'t even exist!')
                raise Exception(
                    'Uh Oh! Looks like the said file path or the numpy array doesn\'t even exist!')
            else:
                lg.info(f'"{desc} array" loaded successsfully!')
                return np.load(open(file_path, 'rb'))
            ...
        except Exception as e:
            lg.exception(e)
            raise e
