import os
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from AWS_ML_Project.constants import *
from AWS_ML_Project.utils.common import read_yaml, create_directories

from AWS_ML_Project import logger

os.chdir('../')

data = pd.read_csv('artifacts/data_ingestion/winequality-red.csv')
# print(data.head())

# By using the below line, you can see the datatypes which we used in the schema.yaml
print(data.info())


@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


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

    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        )

        return data_validation_config


class DataValidation:

    def __init__(self, config: DataValidationConfig):

        self.config = config

    # Here, by comparing data types in the columns in schema.yaml and file we have (wineqaulity-red.csv),
    # we are validating the data type in
    def validate_all_columns(self) -> bool:

        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:

                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e


try:
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_columns()
except Exception as e:
    raise e
