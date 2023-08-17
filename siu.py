import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar mi dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X= dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categoricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",
         OneHotEncoder(categories='auto'),
         [1,2]
         )
        ],
    remainder='passthrough'
    )

X = transformer.fit_transform(X)

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalada de variables
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)

# Parte 2 - construir la RNA
import keras
from keras.models import Sequential
from keras.layers import Dense

# inicializar la RNA
classifier = Sequential()

# agregar capas de entrada y primera capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu", input_dim = 13))

# segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))

# Capa de salida
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar la RNA
classifier.fit(Xtrain, Ytrain, batch_size= 10, epochs=100)

# Prediccion d elos resultados del conjunto de testing
y_pred = classifier.predict(Xtest)
y_pred = (y_pred>0.5)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, y_pred)









