import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import yaml
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import logging
import dagshub
import mlflow 
dagshub.init(repo_owner='Kaushal-1508', repo_name='ML-Pipeline', mlflow=True)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def hyperparameter_tuning(X_train,y_train, param_grid):
    rf= RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf , param_grid=param_grid, cv=3 , n_jobs= -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    return grid_search
## load parameter from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id

def train(data_path, model_path, random_state, n_estimators , max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    remote_server_uri = "https://dagshub.com/Kaushal-1508/ML-Pipeline.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)


    ## start MLFLOW run
    experiment_id = get_or_create_experiment_id("Pima Indians Diabetes Database")
    with mlflow.start_run(experiment_id=experiment_id):
        ## split data into train and test data set
        X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size= 0.2 )
        signature = infer_signature(X_train , y_train)

        ## Define hyperparameter grid 
        
        param_grid = {
            'n_estimators': [100 , 200, 300],
            'max_depth' : [5 ,7, 10 , 11, 15, None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
        }

        grid_search = hyperparameter_tuning(X_train, y_train , param_grid)

        best_model = grid_search.best_estimator_

        ## predict and evaluate the model

        y_pred = best_model.predict(X_test)
        accuracy =accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test , y_pred)
        print(f" Accuracy : {accuracy}")
        ## log addition metric
        mlflow.log_metric("accuracy" , accuracy)
        mlflow.log_param("best_n_estimatiors", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'] )
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'] )
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'] )
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr , "classification_report.txt")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, name="model",registered_model_name="best_Model")
        else:
            mlflow.sklearn.log_model(best_model, name="model", signature=signature)
        ## create dir to save model
        os.makedirs(os.path.dirname(model_path) , exist_ok=True)

        filename = model_path
        pickle.dump(best_model , open(filename , 'wb'))

        print(f" model save to {model_path}")

if __name__ == "__main__":

    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])

