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



def prediction(dataframe,
               target='TARGET',
               models=[LinearSVC(),
                       RandomForestClassifier(n_estimators=750),
                       GradientBoostingClassifier(),
                       LogisticRegression()
                      ]
              ):
    Infos={'nombre de lignes':len(dataframe.index),'nombre de colonnes':len(dataframe),'ratio':0.25}
    print("création des échantillons")
    Y=np.array(list(dataframe[target]))
    X=np.array(list(zip([list(dataframe[colonne]) for colonne in dataframe if colonne!=target]))).reshape(-1,len(dataframe.columns)-1)
    train_x, test_x, train_y, test_y  = train_test_split(X, Y, test_size = 0.25, random_state = 44)
    del X,Y
    print("Apprentissage des modèles")
    models_trained=[mod.fit(train_x,train_y) for mod in models]
    del train_x,train_y
    def résultats(mod):
        return {"score":round(mod.score(test_x,test_y)*100,2),"modèle":str(mod)}
    results=[résultats(mod) for mod in models_trained]
    Infos['scores']=results
    del models_trained
    [print("La précision de {} est : {}%".format(res['modèle'],res["score"])) for res in results]
    f=open("results.txt",'a+',encoding="utf-8")
    f.write(str(Infos)+'\n')
    f.close()