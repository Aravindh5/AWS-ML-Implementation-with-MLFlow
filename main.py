from src.AWS_ML_Project import logger
from AWS_ML_Project.pipeline.stage_01_data_ingesion import DataIngestionTrainingPipeline
from AWS_ML_Project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from AWS_ML_Project.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from AWS_ML_Project.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from AWS_ML_Project.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


logger.info('Welcome to our custom logging.')

STAGE_NAME = "DATA INGESTION STAGE"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} has completed <<<<<<<<<<<<\n\nx=====================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "DATA VALIDATION STAGE"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} has completed <<<<<<<<<<<<\n\nx=====================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "DATA TRANSFORMATION STAGE"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<<<<\n\nX====================X")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL TRAINING STAGE"
try:
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} has completed <<<<<<<<<<<\n\nX====================X")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL EVALUATION STAGE"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} <<<<<<<<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} has completed <<<<<<<<<<<<<\n\nX==============X")
except Exception as e:
    logger.exception(e)
    raise e
