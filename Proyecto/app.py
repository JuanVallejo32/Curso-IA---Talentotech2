from flask import Flask, render_template, request
import pickle
import numpy as np  # Asegúrate de importar numpy

# Inicializar mi aplicación de Flask
app = Flask(__name__)

# Cargar el modelo de café
with open('modelo_cafe.pkl', 'rb') as f:
    modelo_cafe = pickle.load(f)

# Cargar el modelo de cacao
with open('modelo_cacao.pkl', 'rb') as f:
    modelo_cacao = pickle.load(f)

# Cargar el modelo de banano
with open('modelo_banano.pkl', 'rb') as f:
    modelo_banano = pickle.load(f)

# Ruta principal
@app.route("/")
def home():
    return render_template("index.html", prediccion=None)

# Café
@app.route("/predecir_cafe", methods=['POST'])
def predecir_cafe():
    area = float(request.form['Area'])
    production = float(request.form['Production'])
    
    datos = np.array([[area, production]])
    prediccion = modelo_cafe.predict(datos)
        
    return render_template("index.html", prediccion=f'El rendimiento para café es: {prediccion[0]:.2f}')

# Cacao
@app.route("/predecir_cacao", methods=['POST'])
def predecir_cacao():
    area = float(request.form['Area'])
    production = float(request.form['Production'])
    
    datos = np.array([[area, production]])
    prediccion = modelo_cacao.predict(datos)
        
    return render_template("index.html", prediccion=f'El rendimiento para cacao es: {prediccion[0]:.2f}')

# Banano
@app.route("/predecir_banano", methods=['POST'])
def predecir_banano():
    area = float(request.form['Area'])
    production = float(request.form['Production'])
    
    datos = np.array([[area, production]])
    prediccion = modelo_banano.predict(datos)
        
    return render_template("index.html", prediccion=f'El rendimiento para banano es: {prediccion[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
