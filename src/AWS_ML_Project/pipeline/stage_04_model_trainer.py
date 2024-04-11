from AWS_ML_Project import logger
from AWS_ML_Project.components.model_trainer import ModelTrainer
from AWS_ML_Project.config.configuration import ConfigurationManager


STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>> Stage {STAGE_NAME} Model Training has been completed")
    except Exception as e:
        logger.exception(e)
        raise e
