import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', delim_whitespace=True,header=None)
features = data.values[:,0:13]
output = data.values[:,13]

def model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, init ='normal', activation='relu'))
    model.add(Dense(6, init='normal', activation = 'relu'))
    model.add(Dense(1, init='normal'))
    model.compile(loss='mean_squared_error', optimizer ='adam')
    return model


estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n=len(features), n_folds=10, random_state=7)
results = cross_val_score(estimator, features, output, cv=kfold)

print results.mean(), results.std()
