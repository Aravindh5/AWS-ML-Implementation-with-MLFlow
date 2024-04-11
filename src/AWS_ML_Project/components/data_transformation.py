import os
import pandas as pd
from AWS_ML_Project import logger
from sklearn.model_selection import train_test_split
from AWS_ML_Project.entity.config_entity import DataTransformationConfig


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
