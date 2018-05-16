from random import randint

# Create 200 random points
from sklearn import svm

d1 = [[randint(-20, 20), randint(-20, 20)] for i in range(200)]

# Classify them as 1 if they are in the circle and 0 if not
result = [(x ** 2 + y ** 2) < 144 and 1 or 0 for (x, y) in d1]

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(d1, result)

print("Classifying input (2,2) as: {:s}".format(str(rbf_svc.predict([[2, 2]]))))
print("Classifying input (14,13) as: {:s}".format(str(rbf_svc.predict([[14, 13]]))))
print("Classifying input (-18,0) as: {:s}".format(str(rbf_svc.predict([[-18, 0]]))))
