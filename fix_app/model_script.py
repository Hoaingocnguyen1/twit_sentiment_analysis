

import sys 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from sqlalchemy import create_engine,text
import pandas as pd

from torch.utils.data import Dataset
from src.models.transformer_manager import ModelManager
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import logging
import mlflow.pyfunc
from transformers import TextClassificationPipeline
import torch
from src.database.baseDatabase import BlobStorage
import requests


from typing import Dict, Any

logger = logging.getLogger(__name__)

load_dotenv(f'{BASE_DIR}/.env')

tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

blob_storage = BlobStorage(os.getenv('LAKE_STORAGE_CONN_STR'))

MLFLOW_MODEL_NAME = os.getenv('MODEL_NAME')

db_conn_string = os.getenv('PGSQL_CONN_STR')


FASTAPI_RELOAD_ENDPOINT = "http://localhost:8000/reload-model"


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = texts
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        
        # Add labels to the dictionary
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Ensure labels are also tensors
        
        return item
    
class TransformerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def predict(self, context, model_input: pd.Series | pd.DataFrame):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.tolist()
        else:
            try:
                texts = list(model_input)
            except:
                logger.error(f"Can't infer: Expected DataFrame, Series or list object, got {type(model_input)} instead.")
                return 
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=False)
        preds = pipe(texts)
        try:
            label_indices = [int(pred["label"].split("_")[-1]) for pred in preds]
        except Exception as e:
            logger.error(f"Error parsing prediction output: {e}")
            raise ValueError("Unexpected prediction format.")

        return label_indices
        


def get_data_from_database(conn_string, query):
    '''
    get data from pgsql
    '''
    engine = create_engine(conn_string)
    with engine.connect() as connection:

        df = pd.read_sql_query(sql=query, con=connection)

    return df

def train_model(df, model_dir='mlflow/artifact/models/modernBERT', train_params: dict | None = None):

    if df.empty:
        logger.warning("No data was acquired. skipping this.")
        return {"run_id": None, 'manager': None, 'f1_score': -1}
    
    text = df['content'].tolist()
    label = df['predicted_sentiment'].tolist()

    text_train, text_eval, label_train, label_eval = train_test_split(text, label, test_size=0.2)

    experiment_name = "SentimentAnalysis"
    try:
        mlflow.create_experiment(experiment_name)  
    except mlflow.exceptions.MlflowException:
        pass

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"twitter_sentiment_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        output_dir = f'{BASE_DIR}/mlflow/artifact/models/{run_id}'
        manager = ModelManager(model_path=f'{BASE_DIR}/{model_dir}', output_dir=output_dir, finetune=True)

        train_encodings = manager.tokenizer(text_train, truncation=True, padding=True, max_length=128)
        eval_encodings = manager.tokenizer(text_eval, truncation=True, padding=True, max_length=128)

        train_data = TextDataset(train_encodings, label_train)
        eval_data = TextDataset(eval_encodings, label_eval)

        logger.info(f"Started training with run ID: {run_id}")
        
        try:
            result = manager.train(train_data, eval_data, **train_params)
        except TypeError as e:
            logger.warning(str(e))
            result = manager.train(train_data, eval_data)
        except Exception as ex:
            logger.error(f"Error while training data: {str(ex)}")
        
        mlflow.log_metric('training_loss', result['final_train_loss'])
        mlflow.log_metrics(result['eval_metrics'])
        mlflow.log_figure(result['confusion_matrix'], artifact_file='figures/confusion_matrix.png')

        manager.save_model(metrics=result['eval_metrics'], cm_fig=result['confusion_matrix'])

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TransformerWrapper(manager.model, manager.tokenizer),
            artifacts={"model_path": output_dir}
        )

    return {"run_id": run.info.run_id, 'manager': manager, 'f1_score': result['eval_metrics']['f1']}

def register_and_promote(train_results: Dict[str, Any], model_name=MLFLOW_MODEL_NAME):

    if train_results['run_id'] is None:
        logger.warning("Result had no run_id. Skipping this step...")
        return

    run_id = train_results['run_id']
    new_f1 = train_results['f1_score']

    client = mlflow.MlflowClient(tracking_uri)
    
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    new_version = result.version

    try:
        current_champion = client.get_model_version_by_alias(model_name, "champion")
        current_f1 = client.get_run(current_champion.run_id).data.metrics.get("f1", 0.0)
    except mlflow.exceptions.RestException:
        current_champion = None
        current_f1 = None

    if current_f1 is None or new_f1 > current_f1:
        # Remove alias from old champion if exists
        if current_champion:
            client.delete_registered_model_alias(model_name, "champion")

        # Promote new model
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=new_version
        )
        logger.info(f"Promoted run {run_id} to champion (f1={new_f1:.4f})")
    else:
        logger.info(f"New model (f1={new_f1:.4f}) not better than current champion (f1={current_f1:.4f})")


def reload_model(fastapi_url='http://localhost:8000/reload-model'):
    try:
        response = requests.post(fastapi_url)
        if response.status_code == 200:
            logger.info("FastAPI service reloaded the model successfully.")
        else:
            logger.warning(f"FastAPI reload failed: {response.text}")
    except Exception as e:
        logger.error(f"Could not notify FastAPI to reload model: {e}")


def main():
    query="""SELECT content, predicted_sentiment FROM "TWEET_STAGING" WHERE created_at >= NOW() - INTERVAL '7 days';"""
    df = get_data_from_database(db_conn_string, query)
    
    train_params = {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-05,
        "scheduler_type": "linear"
    }
    train_results = train_model(df, train_params = train_params)
    register_and_promote(train_results)

if __name__ == "__main__":
    main()
    #reload_model(FASTAPI_RELOAD_ENDPOINT)


    

    




        
        


    


    










