import tensorflow as tf
from sklearn.metrics import classification_report, mean_squared_error
from preprocess import preprocess_data

def evaluate():
    _, _, _, _, X_test_mapped_scaled, y_test = preprocess_data()

    model = tf.keras.models.load_model('models/model.h5')

    y_pred_test = model.predict(X_test_mapped_scaled)
    prob = tf.nn.sigmoid(y_pred_test)

    mse = mean_squared_error(y_test, prob)
    report = classification_report(y_test, prob.numpy().round())

    print(f"MSE: {mse}")
    print(report)

if __name__ == '__main__':
    evaluate()