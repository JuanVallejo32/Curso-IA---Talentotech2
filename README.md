# Curso-IA---Talentotech2

 ==Predicción de producción agrícola==

   >Esta aplicación de predicción agrícola basada en técnicas de inteligencia artificial permite optimizar el rendimiento de 
    los cultivos mediante el análisis de datos históricos de producción, facilitando la toma de decisiones informadas para 
    pequeños y medianos agricultores en Colombia, con el fin de maximizar la eficiencia en el uso de recursos, reducir las 
    pérdidas agrícolas y garantizar la sostenibilidad de la producción.

 ==Integrantes==
   * Elizabeth Rojas Vargas
   * Yonathan Alexis Pérez Ruiz
   * Henry Asdrúbal Rodríguez Morales
   * Juan Sebastián Vallejo Henao
   * Mauricio Escobar Gutiérrez

 ==Instrucciones==


 ==Codigo==

 ```
#Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#Leer y crear una copia del dataset
data1 = pd.read_excel('Cacao.xlsx')
data = data1.copy()
data

data1.info()

#Se crea un dataframe en la variable data con las columnas a trabajar y se reemplazan los valores 0.0 por NaN
data = data1[['Area (ha)', 'Produccion (ton)','Rendimiento (ha/ton)']]
data.replace(0.0, np.nan, inplace=True)
data

#Separación variables y se eliminan los datos NaN
X = data[['Area (ha)','Produccion (ton)']].dropna()
y = data['Rendimiento (ha/ton)'].dropna()

#Divido datos de entrenamiento
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creacion del modelo
rf_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
#se entrena el modelo
rf_regresion.fit(X_train, Y_train)
#se crea la prediccion
y_pred_rf = rf_regresion.predict(X_test)

#se crea el error medio cuadrado
mse = mean_squared_error(Y_test, y_pred_rf)
# se crea el R2 random forest
r2_rf = r2_score(Y_test, y_pred_rf)

# se imprime los resultados
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2_rf:.2f}')

# se realiza validacion cruzada
cvs_rf = cross_val_score(rf_regresion, X, y, cv=5, scoring='r2')
# se calcula la media de la validacion cruzada
cvs_rf_mean = cvs_rf.mean()
# se calcula la mediana de la validacion cruzada
cvs_rf_median = np.median(cvs_rf)

print(f'Validacion cruzada con R2 en Random Forest: {cvs_rf}')
print(f'Validacion cruzada con R2 con la media de Random Forest: {cvs_rf_mean}')
print(f'Validacion cruzada con R2 con la mediana de Random Forest: {cvs_rf_median}')

#se grafica
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train['Area (ha)'], Y_train, c=Y_train.values.ravel(), cmap='viridis')

#linea de codigo donde se genera un vertice pero se descuadra todo el grafico
#ax.plot([x_test.min(), x_test.max()], [x_test.min(), x_test.max()], 'k--', lw=1)

ax.set_xlabel('Area (ha)')
ax.set_ylabel('Rendimiento (ha/ton)')
ax.set_title("Dataset Visualization")
plt.show()
```
