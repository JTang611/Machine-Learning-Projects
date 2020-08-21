from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

def map_float(x):
    return [float(i) for i in x]

f, n = map(int, input().split())
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

train = [map_float(input().split()) for i in range(n)]
X = [row[:-1] for row in train]
y = [row[-1:] for row in train]
model.fit(X, y)

n = int(input())
X_pred = [map_float(input().split()) for i in range(n)]
for i in model.predict(X_pred):
    print(i[0])
