
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

# Read Data
url = 'https://goo.gl/sXleFv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, delim_whitespace=True, names=names)

# Feature - Target Split
data_values = data.values
print(data.shape)
feature = data_values[:,0:13]
target = data_values[:,13]


# Models
models = []
models.append(('LinearRegression',LinearRegression()))
models.append(('Ridge',Ridge()))
models.append(('Lasso',Lasso()))
models.append(('ElasticNet',ElasticNet()))

# Cross Validation
results =[]
names =[]
for name, model in models:
	kfold = KFold(n_splits=10, random_state = 7)
	cv_results = cross_val_score(model, feature, target, cv=kfold , scoring = "r2")
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()