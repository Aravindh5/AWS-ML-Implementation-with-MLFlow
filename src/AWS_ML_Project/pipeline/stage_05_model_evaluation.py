import os
from AWS_ML_Project import logger
from AWS_ML_Project.config.configuration import ConfigurationManager
from AWS_ML_Project.components.model_evaluation import ModelEvaluation
from dotenv import load_dotenv


load_dotenv()
STAGE_NAME = 'Model Evaluation Stage'

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} <<<<<<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<\n\nX==============X")
    except Exception as e:
        logger.exception(e)
        raise e
