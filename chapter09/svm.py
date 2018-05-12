from sklearn import svm

X = [[1, 0, 1], [-1, 0, -1]]
y = [1, -1]
clf = svm.LinearSVC()
clf.fit(X, y)

print("Predicted value: {:s}".format(str(clf.predict([[1, 1, 1]]))))
