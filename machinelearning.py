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
from sklearn.metrics import accuracy_score, classification_report
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import eli5
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
import shap
import pandas as pd


def score_personnalisé(prediction,realite):
    VP_FP_FN=list(filter(lambda x : (x[0]==0 and x[1]==0)==False ,zip(prediction,realite)))
    #print("Taille de la liste :",len(realite))
    #print("Taille de la liste sans les vrais négatifs:",len(VP_FP_FN))
    VP=len(list(filter(lambda x : x[0]==x[1] ,VP_FP_FN)))
    return round(VP*100/len(VP_FP_FN),2)

def prediction(dataframe,
               target='TARGET',
               models=[{"modèle":LinearSVC,"paramètres":{"random_state":44}}
                      ],
               dataframe_non_qualifié=None
              ):
    print("création des échantillons")
    X_=dataframe_non_qualifié
    Y=np.array(list(dataframe[target]))
    X=np.array(list(zip([list(dataframe[colonne]) for colonne in dataframe if colonne!=target]))).reshape(-1,len(dataframe.columns)-1)

    print("Qualification des données...")
    X_=np.array(X_).reshape(-1,len(X_.columns))
    train_x, test_x, train_y, test_y  = train_test_split(X, Y, test_size = 0.20, random_state = 44)
    del X,Y
    print("Apprentissage des modèles")
    for mod in models:
        mlflowtisation(train_x,train_y,test_x,test_y,
                       modele=[mod['modèle']],
                       params=mod['paramètres'],
                       nombre_de_lignes=len(dataframe.index),
                       nombre_de_colonnes=len(dataframe.columns)-1,
                       dataframe_non_qualifié=dataframe_non_qualifié
                       )


    
def mlflowtisation(
                   train_x,train_y,test_x,test_y,
                   modele=[ElasticNet],
                   params={"random_state":44},
                   nombre_de_lignes="",
                   nombre_de_colonnes="",
                   dataframe_non_qualifié=None
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
        try:
            with open(type(mod).__name__+'.html', 'w') as f:
                f.write(str(eli5.show_weights(mod).data))
        except Exception as e:
            print(e)
        try:
            shap.initjs()
            explainer = shap.TreeExplainer(mod)
            observations = mod.transform(train_x.sample(1000))
            shap_values = explainer.shap_values(observations)
            i = 0
            shap.force_plot(explainer.expected_value, shap_values[i], features=observations[i])
        except Exception as e:
            print(e)
        predicted_qualities = np.array(list(map(lambda x: categ(x),list(mod.predict(test_x)))))
        acc = eval_metrics(test_y, predicted_qualities)
        rapport_details=classification_report(test_y, predicted_qualities)
        f=open(type(mod).__name__+'.txt', 'w')
        f.write(rapport_details)
        f.close()
        print("La précision du modèle {} est : {}%".format(str(mod),round(acc*100,2)))
        
        try:
            print("Qualification des données...")
            res=mod.predict(dataframe_non_qualifié)
            pd.DataFrame.from_dict({"pred":list(res)}).to_csv(type(mod).__name__+'.csv')
        except:
            pass
        mlflow.log_param("Modèle utilisé", type(mod).__name__)
        for param in params:
            mlflow.log_param(param, params[param])
        mlflow.log_param("nombre de lignes", nombre_de_lignes)
        mlflow.log_param("nombre de colonnes", nombre_de_colonnes)
        mlflow.log_param("rapport_details", rapport_details)
        mlflow.log_metric("acc", acc) 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(mod,"model", registered_model_name=str(mod))
        else:

            mlflow.sklearn.log_model(mod,"model")

            mlflow.sklearn.log_model(mod, "model")
    os.chdir(str(path))

