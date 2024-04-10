from src.AWS_ML_Project import logger
from AWS_ML_Project.pipeline.stage_01_data_ingesion import DataIngestionTrainingPipeline

logger.info('Welcome to our custom logging.')
STAGE_NAME = "DATA INGESTION STAGE"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} data ingestion has completed <<<<<<<<<<<<\n\nx=====================x")
except Exception as e:
    logger.exception(e)
    raise e
