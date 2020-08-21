# Thsi file gives a linear regression model to predict future house prices based on input previous data.
from sklearn import linear_model
# load input
f, n = input().split()
f = int(f)
n = int(n)
# train_test_split
clf = linear_model.LinearRegression()
x_train = []
y_train = []

for i in range(n):
    tmp = [float(n) for n in input().split()]
    x_train.append(tmp[0: len(tmp) - 1])
    y_train.append(tmp[len(tmp) - 1])
# train the GLM model
clf.fit(x_train, y_train)

x_test = []
n = int(input())
for i in range(n):
    tmp = [float(n) for n in input().split()]
    x_test.append(tmp)
# predict with test input
y_test = clf.predict(x_test)
for y in y_test:
    print(y)
