import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def preprocess_data():
    df = pd.read_csv('data/cancer_classification.csv', sep=',')

    df_correlacao = df.corr().iloc[:, -1:]
    features_com_corr = df_correlacao.loc[(df_correlacao['benign_0__mal_1'] >= 0.5) | (df_correlacao['benign_0__mal_1'] <= -0.5)].iloc[:-1, :]
    feature_list = list(features_com_corr.index)

    x = df[feature_list]
    y = df['benign_0__mal_1']

    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler_poly = StandardScaler()

    # Add polynomial features and scale the train set
    X_train_mapped = poly.fit_transform(x_train)
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

    # Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Add polynomial features and scale the test set
    X_test_mapped = poly.transform(x_test)
    X_test_mapped_scaled = scaler_poly.transform(X_test_mapped)

    return X_train_mapped_scaled, y_train, X_cv_mapped_scaled, y_cv, X_test_mapped_scaled, y_test

if __name__ == '__main__':
    preprocess_data()