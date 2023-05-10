from flask import Flask, request, Response
import pickle
import tensorflow as tf
from keras.optimizers import Adam


#pip install -U flask-cors
from flask_cors import CORS

application = Flask(__name__)
CORS(application)

with open('dict_vectorizer_coches_net.pck', 'rb') as f:
  dv = pickle.load(f)
model = tf.keras.models.load_model('modelo_keras_coches_net.hdf5')
# Crear un nuevo optimizador
optimizer = Adam(learning_rate=0.01)

# Compilar el modelo con el nuevo optimizador
model.compile(loss='mean_squared_error', optimizer=optimizer)

@application.route('/flask', methods=['GET'])
def flask():
    return '<p>https://flask.palletsprojects.com/en/2.2.x/</p>'

@application.get('/api/get')
def get_method():
    word = request.args.get('word', '<no word>')
    return {
        'hello': 'hello, ' + word
    }
    
@application.post('/api/predict/coche')
def predict_car():
    coche = request.get_json()
    for key, value in coche.items():
        if key in ('km', 'cubicCapacity', 'hp', 'doors', 'year'):
            coche[key] = float(value)
    print(coche)
    if coche is not None:
        print('data:',[dict(coche)])
        X_train = dv.transform([dict(coche)])
        precio = model.predict(X_train)
        print('precio', precio)
        return {'precio': round(float(precio[0]), 2)}
    else:
        return 'No valido!'

@application.route('/')
def main():
    return '<p>Hello, World!</p>'

application.run()