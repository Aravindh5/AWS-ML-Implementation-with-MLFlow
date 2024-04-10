from AWS_ML_Project import logger
from AWS_ML_Project.config.configuration import ConfigurationManager
from AWS_ML_Project.components.data_ingestion import DataIngestion

STAGE_NAME = "DATA INGESTION STAGE"


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':

    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} data ingestion has completed <<<<<<<<<<<<\n\nx=====================x")
    except Exception as e:
        logger.exception(e)
        raise e
