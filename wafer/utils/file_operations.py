import os
import json
import pickle
from wafer.logger import lg
import pandas as pd
import numpy as np
from typing import Dict


class BasicUtils:

    @classmethod
    def read_json_file(cls, file_path: str, file_desc: str) -> Dict:
        try:
            lg.info(f"fetching the data from the \"{file_desc}\" lying at \"{file_path}\"..")
            with open(file_path, 'r') as f:
                data = json.load(f)
                lg.info("data fetched successfully!")
            f.close()

            return data
            ...
        except Exception as e:
            lg.exception(e)