import math

import matplotlib.pyplot as plt


class matchrow:
    def __init__(self, row, allnum=False):
        if allnum:
            self.data = [float(row[i]) for i in range(len(row) - 1)]
        else:
            self.data = row[0:len(row) - 1]
            self.match = int(row[len(row) - 1])


def loadmatch(f, allnum=False):
    return [matchrow(line.split(','), allnum) for line in open(f, 'r')]


def plotagematches(rows):
    xdmatch, ydmatch = [float(r.data[0]) for r in rows if r.match == 1], [float(r.data[1]) for r in rows if r.match == 1]
    xdnomatch, ydnomatch = [float(r.data[0]) for r in rows if r.match == 0], [float(r.data[1]) for r in rows if r.match == 0]

    plt.scatter(xdmatch, ydmatch, color='blue', label='Match Age')
    plt.scatter(xdnomatch, ydnomatch, color='red', label='No Match Age')

    avgs = lineartrain(rows)
    plt.scatter(avgs[0][0], avgs[0][1], color='grey', label='No Match Average')
    plt.scatter(avgs[1][0], avgs[1][1], color='black', label='Match Average')

    plt.show()


def lineartrain(rows):
    averages = {}
    counts = {}

    for row in rows:
        # Get the class of this point
        cl = row.match

        averages.setdefault(cl, [0.0] * (len(row.data)))
        counts.setdefault(cl, 0)

        # Add this point to the averages
        for i in range(len(row.data)):
            averages[cl][i] += float(row.data[i])

        # Keep track of how many points in each class
        counts[cl] += 1

    # Divide sums by counts to get the averages
    for cl, avg in averages.items():
        for i in range(len(avg)):
            avg[i] /= counts[cl]

    return averages


def dotproduct(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def dpclassify(point, avgs):
    # Calculating C, which is the average of M0 and M1, or (M0 +M1)/2
    c = (dotproduct(avgs[1], avgs[1]) - dotproduct(avgs[0], avgs[0])) / 2

    # class=sign(X.M0 – X.M1 + (M0.M0 – M1.M1)/2)
    clazz = dotproduct(point, avgs[0]) - dotproduct(point, avgs[1]) + c

    if clazz > 0:
        return 0
    else:
        return 1


def yesno(v):
    if v == 'yes':
        return 1
    elif v == 'no':
        return -1
    else:
        return 0


def matchcount(interest1, interest2):
    l1 = interest1.split(':')
    l2 = interest2.split(':')
    x = 0
    for v in l1:
        if v in l2:
            x += 1
    return x


def milesdistance(a1, a2):
    return 0


# def milesdistance(a1,a2):
#     lat1,long1=getlocation(a1)
#     lat2,long2=getlocation(a2)
#     latdif=69.1*(lat2-lat1)
#     longdif=53.0*(long2-long1)
#     return (latdif**2+longdif**2)**.5

def loadnumerical():
    oldrows = loadmatch('matchmaker.csv')
    newrows = []
    for row in oldrows:
        d = row.data
        data = [float(d[0]), yesno(d[1]), yesno(d[2]), float(d[5]), yesno(d[6]), yesno(d[7]), matchcount(d[3], d[8]), milesdistance(d[4], d[9]), row.match]
        newrows.append(matchrow(data))
    return newrows


def scaledata(rows):
    # 999999999.0 is Double.MAX_VALUE and -999999999.0 is Double.MIN_VALUE
    low = [999999999.0] * len(rows[0].data)
    high = [-999999999.0] * len(rows[0].data)

    # Find the lowest and highest values
    for row in rows:
        d = row.data
        for i in range(len(d)):
            if d[i] < low[i]:
                low[i] = d[i]
            if d[i] > high[i]:
                high[i] = d[i]

    # Create a function that scales data
    def scaleinput(d):
        return [(d[i] - low[i]) / (high[i] - low[i]) for i in range(len(low)) if high[i] - low[i] != 0]

    # Scale all the data
    newrows = [matchrow(scaleinput(row.data) + [row.match]) for row in rows]

    # Return the new data and the function
    return newrows, scaleinput


# The radial-basis function is like the dot-product in that it takes two vectors and returns a value.
def rbf(v1, v2, gamma=20):
    dv = [float(v1[i]) - float(v2[i]) for i in range(len(v1))]
    l = veclength(dv)
    return math.e ** (-gamma * l)


def veclength(v):
    return sum([p ** 2 for p in v])


def nlclassify(point, rows, offset, gamma=10):
    sum0 = 0.0
    sum1 = 0.0
    count0 = 0
    count1 = 0

    for row in rows:
        if row.match == 0:
            sum0 += rbf(point, row.data, gamma)
            count0 += 1
        else:
            sum1 += rbf(point, row.data, gamma)
            count1 += 1

    y = (1.0 / count0) * sum0 - (1.0 / count1) * sum1 + offset

    if y > 0:
        return 0
    else:
        return 1


def getoffset(rows, gamma=10):
    l0 = []
    l1 = []

    for row in rows:
        if row.match == 0:
            l0.append(row.data)
        else:
            l1.append(row.data)

    sum0 = sum(sum([rbf(v1, v2, gamma) for v1 in l0]) for v2 in l0)
    sum1 = sum(sum([rbf(v1, v2, gamma) for v1 in l1]) for v2 in l1)
    return (1.0 / (len(l1) ** 2)) * sum1 - (1.0 / (len(l0) ** 2)) * sum0
