import math

import optimization


def costf(x): return (1.0 / (x[0] + 0.1)) * math.sin(x[0])


domain = [(0, 20)]
print("{:s}".format(str(optimization.annealingoptimize(domain, costf))))
