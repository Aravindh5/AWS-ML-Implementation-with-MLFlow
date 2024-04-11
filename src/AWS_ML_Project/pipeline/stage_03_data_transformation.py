# import os
# import pandas as pd
from pathlib import Path
# from AWS_ML_Project import logger
# from sklearn.model_selection import train_test_split
from AWS_ML_Project.config.configuration import ConfigurationManager
from AWS_ML_Project.components.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation Stage."


class DataTransformationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        # If our data is not in the correct format, then our pipeline will breach in stage 03 of data transformation.
        # Previously, while we were doing the data validation, we kept one file in the name of "status.txt".
        # So, before doing the transformation, we are checking this file that whether the data validation has done.
        # If Status is True, then the flow of the pipeline will continue to Stage 03 of Data Transformation.
        # Otherwise, the pipeline will stop in Stage 03.
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status:
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting()
            else:
                raise Exception("Your Data Scheme is not valid.")

        except Exception as e:
            raise e
