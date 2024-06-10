import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from azureml.core import Run

from preprocess import preprocess_data

def train():
    X_train_mapped_scaled, y_train, _, _, _, _ = preprocess_data()

    model = Sequential(
        [
            Dense(15, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')
        ]
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy', 'mse']
    )

    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=100
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/model.h5')

    # Registro do modelo no Azure ML
    run = Run.get_context()
    run.upload_file(name='models/model.h5', path_or_stream='models/model.h5')
    run.register_model(model_name='my_model', model_path='models/model.h5')

if __name__ == '__main__':
    train()