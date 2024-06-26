import os
from pathlib import Path
from dataclasses import dataclass

from AWS_ML_Project.constants import *
from AWS_ML_Project.utils.common import read_yaml, create_directories

import zipfile
import urllib.request as request
from AWS_ML_Project import logger
from AWS_ML_Project.utils.common import get_size


os.chdir('../')


@dataclass(frozen=True)
class DataIngestionConfig:

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


class ConfigurationManager:

    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):

        self.config = config

    def download_file(self):

        if not os.path.exists(self.config.local_data_file):

            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file,
            )
            logger.info(f"{filename} download! with following info: \n{headers}")

        else:

            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):

        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None.
        :return:
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


# This is the data ingestion pipeline (phase).
# Based on my actual project, I'll modify this data ingestion phase.
try:
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
except Exception as e:
    raise e
