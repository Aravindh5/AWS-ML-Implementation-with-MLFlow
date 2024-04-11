import os

os.chdir('../')

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Aravindh5/AWS-ML-Implementation-with-MLFlow.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Aravindh5'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '92a8a1192753d800264fc7311c97acd4d54f4d9f'


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str


# Preparation for configuation manager
from AWS_ML_Project.constants import *
from AWS_ML_Project.utils.common import read_yaml, create_directories, save_json


class ConfigurationManager:

    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri='https://dagshub.com/Aravindh5/AWS-ML-Implementation-with-MLFlow.mlflow'
        )

        return model_evaluation_config


# Components
import joblib
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):

        self.config = config

    def eval_metrics(self, actual, pred):

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(actual=test_y, pred=predicted_qualities)

            # Saving Metrics as Local
            scores = {'rmse': rmse, 'mae': mae, 'r2': r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('r2', r2)

            # Model Registry doesn't work with file store
            if tracking_url_type_store != 'file':

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case
                # Please refer to the doc for more information
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, 'model', registered_model_name='ElaticNetModel')

            else:

                mlflow.sklearn.log_model(model, 'model')


try:
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
    model_evaluation_config.log_into_mlflow()
except Exception as e:
    raise e
