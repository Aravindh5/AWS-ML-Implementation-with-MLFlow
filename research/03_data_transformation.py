import os


os.chdir('../')


# Based on README.md, we are following the steps for building data pipelines.
# Step 1: Update config.yaml (Data Transformation). Below, add the Data Transformation Config
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataTransformationConfig:

    root_dir: Path
    data_path: Path


# Step 2:

# Step 5: Updating Configuration Manager.
from AWS_ML_Project.constants import *
from AWS_ML_Project.utils.common import read_yaml, create_directories


class ConfigurationManager:

    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH,
            schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config


# Step 6: Updating the Components
# Here, we did necessary data science stuffs like EDA, Analysis and other stuffs.
# But, here I am only creating a necessary ml pipeline and ml flow stuffs.
import os
from AWS_ML_Project import logger
from sklearn.model_selection import train_test_split
import pandas as pd


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):

        self.config = config


    # Note: Here, you can add different data transformation techniques such as Scaler, PCA and all.
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model.

    # Here, I am only adding train_test_split because we have cleaned up data.


    def train_test_splitting(self):

        data = pd.read_csv(self.config.data_path)

        # Split the data into train and test sets. (0.75, 0.25)
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and testing sets")
        logger.info(f"Train Data Shape : {train.shape}")
        logger.info(f"Test Data Shape: {test.shape}")


try:
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.train_test_splitting()
except Exception as e:
    raise e
