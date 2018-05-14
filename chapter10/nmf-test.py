import numpy as np
import nmf

l1 = [[1, 2, 3], [4, 5, 6]]
print(l1)

m1 = np.matrix(l1)
print("m1:\n {:s}".format(str(m1)))

m2 = np.matrix([[1, 2], [3, 4], [5, 6]])
print("m2:\n {:s}".format(str(m2)))

print("m1 * m2:\n {:s}".format(str(m1 * m2)))

print("Shape m1: {:s}".format(str(np.shape(m1))))
print("Shape m2: {:s}".format(str(np.shape(m2))))

a1 = m1.A
print("Fast array from m2:\n {:s}".format(str(a1)))

w, h = nmf.factorize(m1 * m2, pc=3, iter=100)
print("Factorized: w * h:\n {:s}".format(str(w * h)))
