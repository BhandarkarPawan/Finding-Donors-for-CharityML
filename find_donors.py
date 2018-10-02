import numpy as np
import pandas as pd
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
import visuals as vs

X_train_OHE = pd.read_pickle('X_train_OHE')
X_test_OHE = pd.read_pickle('X_test_OHE')
y_train_OHE = pd.read_pickle('y_train_OHE')
y_test_OHE = pd.read_pickle('y_test_OHE')

X_train_LE = pd.read_pickle('X_train_LE')
X_test_LE = pd.read_pickle('X_test_LE')
y_test_LE = pd.read_pickle('y_test_LE')
y_train_LE = pd.read_pickle('y_train_LE')


model = AdaBoostClassifier(random_state=0)
model = model.fit(X_train_LE, y_train_LE)
y_pred_LE = model.predict(X_test_LE)
LE_score_acc = accuracy_score(y_test_LE, y_pred_LE)
LE_score_fbeta = fbeta_score(y_test_LE, y_pred_LE, beta=0.5)
print("LE Acc: ", LE_score_acc)
print("LE f: ", LE_score_fbeta)

model.fit(X_train_OHE, y_train_OHE)
y_pred_OHE = model.predict(X_test_OHE)
OHE_score_acc = accuracy_score(y_test_OHE, y_pred_OHE)
OHE_score_fbeta = fbeta_score(y_test_OHE, y_pred_OHE, beta=0.5)
print("OHE Acc: ", OHE_score_acc)
print("OHE f: ", OHE_score_fbeta)
feature_importannces = model.feature_importances_
vs.feature_plot(feature_importannces, X_train_OHE, y_train_OHE)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
# TODO: Initialize the classifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
t0 = time()
grid_fit = grid_obj.fit(X_train_OHE, y_train_OHE)
print("That took {} seconds".format(time() - t0))

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and modelfk
predictions = (clf.fit(X_train_OHE, y_train_OHE)).predict(X_test_OHE)
best_predictions = best_clf.predict(X_test_OHE)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test_OHE, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test_OHE, predictions, beta=0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(
    accuracy_score(y_test_OHE, best_predictions)))
print(
    "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test_OHE, best_predictions, beta=0.5)))
