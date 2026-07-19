from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model   = joblib.load('modelo_regresion_lineal.pkl')
scaler  = joblib.load('escalador.pkl')

# Las 13 features seleccionadas (mismo orden que en entrenamiento)
FEATURES = [
    'alto_cm', 'ancho_cm', 'anio_creacion',
    'categoria_Cerámica', 'categoria_Escultura', 'categoria_Fotografía',
    'categoria_Ilustración', 'categoria_Pintura',
    'con_certificado', 'disponible_envio', 'es_original',
    'material_Papel fotográfico', 'material_Tela'
]

def build_features(data):
    categoria = data.get('categoria', '')
    material  = data.get('material', '')

    row = {
        'alto_cm'                   : float(data.get('alto_cm', 0)),
        'ancho_cm'                  : float(data.get('ancho_cm', 0)),
        'anio_creacion'             : float(data.get('anio_creacion', 2020)),
        'categoria_Cerámica'        : 1 if categoria == 'Cerámica'   else 0,
        'categoria_Escultura'       : 1 if categoria == 'Escultura'  else 0,
        'categoria_Fotografía'      : 1 if categoria == 'Fotografía' else 0,
        'categoria_Ilustración'     : 1 if categoria == 'Ilustración'else 0,
        'categoria_Pintura'         : 1 if categoria == 'Pintura'    else 0,
        'con_certificado'           : int(data.get('con_certificado', 0)),
        'disponible_envio'          : int(data.get('disponible_envio', 0)),
        'es_original'               : int(data.get('es_original', 1)),
        'material_Papel fotográfico': 1 if material == 'Papel fotográfico' else 0,
        'material_Tela'             : 1 if material == 'Tela'              else 0,
    }
    return np.array([[row[f] for f in FEATURES]])

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'api': 'NU★B Studio — Predictor de Precio',
        'version': '1.0',
        'endpoints': {
            'GET  /health': 'Estado del servidor',
            'POST /predecir': 'Predice el precio de una obra'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Se requiere JSON con los datos de la obra'}), 400

        X = build_features(data)
        X_scaled = scaler.transform(X)
        precio = float(model.predict(X_scaled)[0])
        precio = max(350, round(precio / 50) * 50)

        return jsonify({
            'precio_predicho': precio,
            'moneda': 'MXN'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
