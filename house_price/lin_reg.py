from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# load Data
boston = load_boston()

# Linear Regression
train_data, test_data, train_target, test_target = train_test_split(boston.data,boston.target,test_size =0.4, random_state=2)
lin_reg = LinearRegression()
lin_reg.fit(train_data,train_target)
coef_val = pd.DataFrame()
coef_val['Attributes'] = boston.feature_names
coef_val['Values'] = lin_reg.coef_
print coef_val

# Prediction on train_data
pred_target = lin_reg.predict(train_data)
plt.scatter(train_target,pred_target)
plt.show()
mean_error = np.mean((train_target-pred_target)**2)
print "Error = ", mean_error, "%"

# Prediction on test_data
pred_target = lin_reg.predict(test_data)
plt.scatter(test_target,pred_target)
plt.show()
mean_error = np.mean((test_target-pred_target)**2)
print "Error = ", mean_error, "%"

