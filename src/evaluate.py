import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse
import logging
import dagshub
import mlflow 
dagshub.init(repo_owner='Kaushal-1508', repo_name='ML-Pipeline', mlflow=True)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

## load parameter from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    remote_server_uri = "https://dagshub.com/Kaushal-1508/ML-Pipeline.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    ## load the save model
    model = pickle.load(open(model_path, 'rb'))
    predictions =  model.predict(X)
    accuracy = accuracy_score(y , predictions)

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy :{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])
