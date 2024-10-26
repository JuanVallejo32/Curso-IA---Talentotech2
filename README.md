# Curso-IA---Talentotech2

 ## Predicción de producción agrícola

   >Esta aplicación de predicción agrícola, basada en técnicas de inteligencia artificial, permite optimizar el rendimiento de los cultivos mediante el análisis de datos históricos de producción. Su objetivo es facilitar la toma de decisiones informadas para pequeños y medianos agricultores en Colombia, maximizando la eficiencia en el uso de recursos, reduciendo las pérdidas agrícolas y promoviendo la sostenibilidad de la producción.

  ## Integrantes
   * Elizabeth Rojas Vargas
   * Yonathan Alexis Pérez Ruiz
   * Henry Asdrúbal Rodríguez Morales
   * Juan Sebastián Vallejo Henao
   * Mauricio Escobar Gutiérrez

  ## Instrucciones
> A continuación encontrará las instrucciones para instalar las dependencias y ejecutar el proyecto en su entorno local:

1. **Clona el repositorio:**
```bash
git clone https://github.com/JuanVallejo32/Curso-IA---Talentotech2.git
cd Curso-IA---Talentotech2
```

2. **Activa el entorno virtual:**
```bash
python -m venv env
source env/bin/activate  o si  usas Windows: `env\Scripts\activate`
```

3. **Instalar las dependencias:**
```bash
pip install requirements.txt
```

 4. **Para iniciar la aplicación:**

 ```bash
 python app.py
 ```

## Código

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

#Información general del dataset

data1.info()

#Creación del DataFrame y reemplazo de valores 0.0 por NaN

data = data1[['Area (ha)', 'Produccion (ton)', 'Rendimiento (ha/ton)']]
data.replace(0.0, np.nan, inplace=True)

#Separación de variables y eliminación de datos NaN

X = data[['Area (ha)', 'Produccion (ton)']].dropna()
y = data['Rendimiento (ha/ton)'].dropna()

#División de los datos de entrenamiento y prueba

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creación y entrenamiento del modelo

rf_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regresion.fit(X_train, Y_train)

#Predicción

y_pred_rf = rf_regresion.predict(X_test)

#Evaluación del modelo

mse = mean_squared_error(Y_test, y_pred_rf)
r2_rf = r2_score(Y_test, y_pred_rf)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2_rf:.2f}')

#Validación cruzada

cvs_rf = cross_val_score(rf_regresion, X, y, cv=5, scoring='r2')
cvs_rf_mean = cvs_rf.mean()
cvs_rf_median = np.median(cvs_rf)
print(f'Validación cruzada con R2 en Random Forest: {cvs_rf}')
print(f'Media de validación cruzada: {cvs_rf_mean}')
print(f'Mediana de validación cruzada: {cvs_rf_median}')

#Gráfico de datos

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train['Area (ha)'], Y_train, c=Y_train.values.ravel(), cmap='viridis')
ax.set_xlabel('Area (ha)')
ax.set_ylabel('Rendimiento (ha/ton)')
ax.set_title("Dataset Visualization")
plt.show()

```
