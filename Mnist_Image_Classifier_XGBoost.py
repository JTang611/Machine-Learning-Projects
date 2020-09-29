!pip install xgboost

!pip install graphviz

import xgboost
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

print(X.shape)
print(y.shape)

% matplotlib inline

print(digits.data.shape)

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 5,                 # the maximum depth of each tree
    'eta': 0.3,                     # the training step for each iteration
    'silent': 1,                    # logging mode - quiet
    'objective': 'multi:softmax',   # multiclass classification using the softmax objective
    'num_class': 10                 # the number of classes that exist in this datset
}
num_round = 500  # the number of training iterations


bstmodel = xgboost.train(param, dtrain, num_round)

#Save as human readable model
bstmodel.dump_model('dump.raw.txt')


preds = bstmodel.predict(dtest)

preds.shape

preds


from sklearn import metrics
acc = metrics.accuracy_score(y_test, preds)

print('Accuracy: %f' % acc)

# Plotting tree does not work yet
# xgboost.plot_tree(bstmodel, num_trees=2)

xgboost.plot_importance(bstmodel)

# let's try grid search
# https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
# Todo adaptation
from sklearn.model_selection import GridSearchCV

clf = xgb.XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)
