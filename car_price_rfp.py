# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 08:22:51 2023

@author: zzirchandra1
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Import data and check for null values

df = pd.read_csv("D:\\resume prep\\Car-Price-Prediction-master\\car data.csv")
df.describe()
df.info()
df.head(5)
df.isnull().sum()


# data Preprocessing
dset = pd.get_dummies(df,columns=['Fuel_Type', 'Seller_Type', 'Transmission'],drop_first=True)
dset['current_year']=2023
dset['years']=dset['current_year']-dset['Year']
dset.drop(columns=dset[['current_year','Year','Car_Name']],inplace=True)


sns.pairplot(dset)
sns.pairplot(dset[['Selling_Price', 'Present_Price', 'Kms_Driven', 'years']])

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X = dset.iloc[:,1:]
y = dset.iloc[:,0]

### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

#Model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV, hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1200, num = 24)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf}

print(random_grid)

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()

rf_randomsearch = RandomizedSearchCV(estimator=RFR, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_randomsearch.fit(X_train,y_train)
print("\n The best estimator across ALL searched params:\n", rf_randomsearch.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_randomsearch.best_score_)
print("\n The best parameters across ALL searched params:\n", rf_randomsearch.best_params_)


predictions=rf_randomsearch.predict(X_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle
# open a file, where you ant to store the data
file = open('D:\\resume prep\\random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_randomsearch, file)

