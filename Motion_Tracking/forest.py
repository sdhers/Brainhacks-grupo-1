# -*- coding: utf-8 -*-
"""
Random forest
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

#%%

# Specify the folder path
ruta = 'DataDLC/videos_eval/'
# folder_path = r'C:\Users\gonza\1\Tesis\2022\practica-PIV\gel8'

# Get the list of file names in the folder
file_names = os.listdir(ruta)

lista_h5 = []
lista_labels = []

# Print the file names
for file_name in file_names:
    if "Position.h5" in file_name:
        lista_h5.append( file_name )
    elif "Labels.csv" in file_name:
        lista_labels.append( file_name )

print(lista_labels)

n_test = 4

D_test = lista_h5.pop(n_test)
l_test = lista_labels.pop(n_test)

#%% Importar datos entrenamiento

ruta = 'DataDLC/videos_eval/'

data = pd.DataFrame(columns = ['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 'neck_x', 'neck_y', 'body_x', 'body_y', 'tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y', 'label'])

for i in range( len( lista_h5 ) ):
    file = ruta + lista_h5[i]
    fileL = ruta + lista_labels[i]

    df = pd.read_hdf(file)['DLC_resnet50_brainhackMay25shuffle1_230000']
    labels = pd.read_csv(fileL)
    
    data0 = pd.DataFrame()
    
    for key in df.keys():
        if key[1] != "likelihood":
            if key[0] == "obj_1" or key[0] == "obj_2":
                data0[str( key[0] )+"_"+str( key[1] )] = [df[key].mean()]*len(df[key])
            else:
                data0[str( key[0] )+"_"+str( key[1] )] = df[key]
        
    data0["label"] = labels["Left"] + 2*labels["Right"]

    data = pd.concat([data,data0], ignore_index=True)

print(data.shape)

#%% Test

ruta = 'DataDLC/videos_eval/'

file = ruta + D_test

df = pd.read_hdf(file)['DLC_resnet50_brainhackMay25shuffle1_230000']
labels = pd.read_csv(ruta + l_test)

test = pd.DataFrame()

for key in df.keys():
    if key[1] != "likelihood":
        test[str( key[0] )+"_"+str( key[1] )] = df[key]
    
test["label"] = labels["Left"] + 2*labels["Right"]

print(test.shape)

#%%


""" Dividimos los datos en entrenamiento y prueba """
# X son nuestras variables independientes
X = data.drop(["label"], axis = 1)

# y es nuestra variable dependiente
y = data.label.astype('int')

# División 75% de datos para entrenamiento, 25% de daatos para test
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
X_train, y_train = X, y

# Creaamos el modelo de Bosques Aleatorios (y configuramos el número de estimadores (árboles de decisión))
BA_model = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = 5, class_weight="balanced")


# random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# grid_forest = {
#     'n_estimators': range(100, 500, 100),
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'max_depth': [3, 4, 5, 8, 10],
#     'max_features': ['sqrt', 'log2', None],
#     'class_weight': ['balanced', None],
# }

BA_model.fit(X_train, y_train)

#%%
import numpy as np
labels = np.array(data.label)

print(labels[[type(l) != type(1) for l in labels]])


#%%

data = pd.read_csv("data.csv")
data = data.drop(['Unnamed: 0'], axis = 1)

""" Dividimos los datos en entrenamiento y prueba """
# X son nuestras variables independientes
X = data.drop(["label"], axis = 1)

# y es nuestra variable dependiente
y = data.label
























