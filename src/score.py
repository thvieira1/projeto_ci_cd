import json
import numpy as np
import tensorflow as tf
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('my_model')
    model = tf.keras.models.load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    return json.dumps(predictions.tolist())