import numpredict

# megapixels, zoom, price
cameras = [{'input': (7.1, 3.8), 'result': 399},
           {'input': (5.0, 2.4), 'result': 299},
           {'input': (6.0, 4.0), 'result': 349},
           {'input': (6.0, 12.0), 'result': 399},
           {'input': (10.0, 3.0), 'result': 449}]

print("Predicted price {:f}".format(numpredict.weightedknn(cameras, (6.0, 6.0), k=3)))

scc = numpredict.rescale(cameras, (1, 2))
print("Rescaled values {:s}".format(str(scc)))


def knn1(d, v):
    return numpredict.knnestimate(d, v, k=1)


print("Cross validation error {:f}".format(numpredict.crossvalidate(knn1, cameras, test=0.3, trials=2)))
print("Cross validation error {:f}".format(numpredict.crossvalidate(knn1, scc, test=0.3, trials=2)))
