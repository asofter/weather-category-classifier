# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns

# Model Selection
'''
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()
'''

# Model Tuning
'''
rfc = RandomForestClassifier(max_features='sqrt') 
 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [9, 18, 27, 36, 45, 54, 63, 100, 150, 200],
           "max_depth" : [1, 3, 5, 10, 15, 20, 25, 30],
           "criterion" : ['gini'],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
'''

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.48, random_state=0)

# Optimized RF classifier
model = RandomForestClassifier(n_estimators=9, max_depth=5, min_samples_leaf=6, random_state=0, criterion='gini')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Conf matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Fisual conf matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=[1,2,3], yticklabels=[1,2,3])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification report
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

joblib.dump(model, 'cache/trained_classifier.pkl') 
