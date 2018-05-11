from random import random

import numpredict
import optimization

print("Predicted wine price: {:5.4f}".format(numpredict.wineprice(95.0, 3.0)))
print("Predicted wine price: {:5.4f}".format(numpredict.wineprice(95.0, 8.0)))
print("Predicted wine price: {:5.4f}".format(numpredict.wineprice(99.0, 1.0)))

data = numpredict.wineset1()

print("Data[0]: {:s}".format(str(data[0])))
print("Data[1]: {:s}".format(str(data[1])))

print("Euclidean distance between data[0] and data[1] is: {:5.4f}".format(numpredict.euclidean(data[0]['input'], data[1]['input'])))

print("kNN estimate for input: (95.0, 3.0) is: {:5.4f}".format(numpredict.knnestimate(data, (95.0, 3.0))))
print("kNN estimate for input: (99.0, 3.0) is: {:5.4f}".format(numpredict.knnestimate(data, (99.0, 3.0))))
print("kNN estimate for input: (99.0, 5.0) is: {:5.4f}".format(numpredict.knnestimate(data, (99.0, 5.0))))

print("The actual price for input: (99.0,5.0) is: {:5.4f}".format(numpredict.wineprice(99.0, 5.0)))
print("kNN(n=1) estimate for input: (99.0, 5.0) is: {:5.4f}".format(numpredict.knnestimate(data, (99.0, 5.0), k=1)))

print("")
print("Weight for distance (inverse) is: {:f}".format(numpredict.inverseweight(0.1)))
print("Weight for distance (subtract) is: {:f}".format(numpredict.subtractweight(0.1)))
print("Weight for distance (gaussian) is: {:f}".format(numpredict.gaussian(1.0)))
print("")
print("Weight for distance (inverse) is: {:f}".format(numpredict.inverseweight(1)))
print("Weight for distance (subtract) is: {:f}".format(numpredict.subtractweight(1)))
print("Weight for distance (gaussian) is: {:f}".format(numpredict.gaussian(3.0)))
print("")
print("Weighted kNN(n=5) estimate for input: (99.0, 5.0) is: {:5.4f}".format(numpredict.weightedknn(data, (99.0, 5.0))))
print("")


def knn3(d, v):
    return numpredict.knnestimate(d, v, k=3)


def knn1(d, v):
    return numpredict.knnestimate(d, v, k=1)


def knninverse(d, v):
    return numpredict.weightedknn(d, v, weightf=numpredict.inverseweight)


print("Cross validation error for weighted gaussian KNN(N=5) is: {:5.4f}".format(numpredict.crossvalidate(numpredict.weightedknn, data)))
print("Cross validation error for weighted inverse KNN(N=5) is: {:5.4f}".format(numpredict.crossvalidate(knninverse, data)))
print("Cross validation error for KNN(N=5) is: {:5.4f}".format(numpredict.crossvalidate(numpredict.knnestimate, data)))
print("Cross validation error for KNN(N=3) is: {:5.4f}".format(numpredict.crossvalidate(knn3, data)))
print("Cross validation error for KNN(N=1) is: {:5.4f}".format(numpredict.crossvalidate(knn1, data)))

data2 = numpredict.wineset2()
print("Cross validation error for KNN(N=3) on dataset 2 is: {:5.4f}".format(numpredict.crossvalidate(knn3, data2)))
print("Cross validation error for weighted gaussian KNN(N=5) on dataset 2 is: {:5.4f}".format(numpredict.crossvalidate(numpredict.weightedknn, data2)))

sdata = numpredict.rescale(data2, [10, 10, 0, 0.5])
print("Cross validation error for KNN(N=3) on rescaled dataset 2 is: {:5.4f}".format(numpredict.crossvalidate(knn3, sdata)))
print("Cross validation error for weighted gaussian KNN(N=5) on rescaled dataset 2 is: {:5.4f}".format(numpredict.crossvalidate(numpredict.weightedknn, sdata)))

print("")

weightdomain = [(0, 20)] * 4
costf = numpredict.createcostfunction(numpredict.knnestimate, data2)
print("Scaled factors calculated with simulated annealing: {:s}".format(str(optimization.annealingoptimize(weightdomain, costf, step=2))))

data3 = numpredict.wineset3()
print("Predicted wine price: {:5.4f}".format(numpredict.wineprice(99.0, 20.0)))
print("Weighted kNN estimate for input: (99.0, 20.0) is: {:5.4f}".format(numpredict.weightedknn(data3, (99.0, 20.0))))
print("Crossvalidation error for weighted gaussian KNN(N=5) on rescaled dataset 3 is: {:5.4f}".format(numpredict.crossvalidate(numpredict.weightedknn, data3)))

print("Guessed probability for distances in range the (40,80) for vector [99,20]: {:5.4f}".format(numpredict.probguess(data, [99, 20], 40, 80)))
print("Guessed Probability for distances in range the (80,120) for vector [99,20]: {:5.4f}".format(numpredict.probguess(data, [99, 20], 80, 120)))
print("Guessed Probability for distances in range the (120,1000) for vector [99,20]: {:5.4f}".format(numpredict.probguess(data, [99, 20], 120, 1000)))
print("Guessed Probability for distances in range the (30,120) for vector [99,20]: {:5.4f}".format(numpredict.probguess(data, [99, 20], 30, 120)))

data = numpredict.wineset3()
sample = data[int(len(data) * random())]['input']
numpredict.cumulativegraph(data, sample, 150)
numpredict.probabilitygraph(data, sample, 150, ss=3)
