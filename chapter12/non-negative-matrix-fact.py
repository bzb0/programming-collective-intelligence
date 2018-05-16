import numpy as np

import nmf

data = np.matrix([[29., 29.],
                  [43., 33.],
                  [15., 25.],
                  [40., 28.],
                  [24., 11.],
                  [29., 29.],
                  [37., 23.],
                  [21., 6.]])

weights, features = nmf.factorize(data, pc=2)
print("{:s}".format(str(weights)))
print("{:s}".format(str(features)))
