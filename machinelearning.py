# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:54:10 2020

@author: minimilien
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
import os




def score_personnalisé(prediction,realite):
    VP_FP_FN=list(filter(lambda x : (x[0]==0 and x[1]==0)==False ,zip(prediction,realite)))
    #print("Taille de la liste :",len(realite))
    #print("Taille de la liste sans les vrais négatifs:",len(VP_FP_FN))
    VP=len(list(filter(lambda x : x[0]==x[1] ,VP_FP_FN)))
    return round(VP*100/len(VP_FP_FN),2)

def prediction(dataframe,
               target='TARGET',
               models=[{"modèle":LinearSVC,"paramètres":{"random_state":44}}
                      ]
              ):
    print("création des échantillons")
    Y=np.array(list(dataframe[target]))
    X=np.array(list(zip([list(dataframe[colonne]) for colonne in dataframe if colonne!=target]))).reshape(-1,len(dataframe.columns)-1)
    train_x, test_x, train_y, test_y  = train_test_split(X, Y, test_size = 0.25, random_state = 44)
    del X,Y
    print("Apprentissage des modèles")
    for mod in models:
        mlflowtisation(train_x,train_y,test_x,test_y,
                       modele=[mod['modèle']],
                       params=mod['paramètres'],
                       nombre_de_lignes=len(dataframe.index),
                       nombre_de_colonnes=len(dataframe.columns)-1
                       )


    
def mlflowtisation(
                   train_x,train_y,test_x,test_y,
                   modele=[ElasticNet],
                   params={"random_state":44},
                   nombre_de_lignes="",
                   nombre_de_colonnes=""
                   ):

    path=os.getcwd()
    os.chdir("./../")

    import logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    def categ(x):
        if x <=0.1:
            return 0
        else:
            return 1
    def eval_metrics(actual, pred):
        acc = accuracy_score(actual, pred)
        return acc
    with mlflow.start_run():
        mod = modele[0](**params)
        mod.fit(train_x, train_y)
        predicted_qualities = np.array(list(map(lambda x: categ(x),list(mod.predict(test_x)))))
        acc = eval_metrics(test_y, predicted_qualities)
        print("La précision du modèle {} est : {}%".format(str(mod),round(acc*100,2)))
        mlflow.log_param("Modèle utilisé", type(mod).__name__)
        for param in params:
            mlflow.log_param(param, params[param])
        mlflow.log_param("nombre de lignes", nombre_de_lignes)
        mlflow.log_param("nombre de colonnes", nombre_de_colonnes)
        mlflow.log_metric("acc", acc)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(mod,"model", registered_model_name=str(mod))
        else:

            mlflow.sklearn.log_model(mod,"model")

            mlflow.sklearn.log_model(mod, "model")
    os.chdir(str(path))

