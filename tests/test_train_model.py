import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import train_model  # Importe seu script de treinamento aqui

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Configuração que será usada em todos os testes
        self.df = pd.read_csv('data/cancer_classification.csv', sep=',')
        self.df_correlacao = self.df.corr().iloc[:,-1:]
        features_com_corr = self.df_correlacao.loc[(self.df_correlacao['benign_0__mal_1'] >= 0.5) | (self.df_correlacao['benign_0__mal_1'] <= -0.5)].iloc[:-1,:]
        self.feature_list = list(features_com_corr.index)
        self.X = self.df[self.feature_list]
        self.y = self.df['benign_0__mal_1']
        
        self.x_train, x_, self.y_train, y_ = train_test_split(self.X, self.y, test_size=0.40, random_state=1)
        self.x_cv, self.x_test, self.y_cv, self.y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler_poly = StandardScaler()

        self.X_train_mapped = self.poly.fit_transform(self.x_train)
        self.X_train_mapped_scaled = self.scaler_poly.fit_transform(self.X_train_mapped)

    def test_data_shapes(self):
        # Verificar se os dados de treinamento foram dimensionados corretamente
        self.assertEqual(self.X_train_mapped_scaled.shape[1], self.X_train_mapped.shape[1])
    
    def test_model_training(self):
        # Verificar se o modelo é treinado corretamente
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

        history = model.fit(
            self.X_train_mapped_scaled, self.y_train,
            epochs=10, verbose=0  # Usar menos épocas para testes rápidos
        )
        
        # Verificar se o treinamento foi realizado
        self.assertTrue(len(history.history['loss']) > 0)

    def test_model_evaluation(self):
        # Verificar se o modelo é avaliado corretamente
        X_cv_mapped = self.poly.transform(self.x_cv)
        X_cv_mapped_scaled = self.scaler_poly.transform(X_cv_mapped)
        
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
        model.fit(self.X_train_mapped_scaled, self.y_train, epochs=10, verbose=0)

        loss, accuracy, mse = model.evaluate(X_cv_mapped_scaled, self.y_cv, verbose=0)
        
        # Verificar se a avaliação foi realizada
        self.assertIsNotNone(loss)
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(mse)

if __name__ == '__main__':
    unittest.main()
