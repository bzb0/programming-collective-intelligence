import clusters


def euclidean(v1, v2):
    return sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])


data = [[1.0, 8.0], [3.0, 8.0], [2.0, 6.0], [1.5, 1.0], [4.0, 2.0]]
labels = ['A', 'B', 'C', 'D', 'E']

hcl = clusters.hcluster(data, distance=euclidean)
kcl = clusters.kcluster(data, distance=euclidean, k=2)

for c in kcl:
    print("{:s}".format(str([labels[l] for l in c])))

clusters.drawdendrogram(hcl, labels, jpeg='hcl.jpg')

labels = ['A', 'B', 'C', 'D']
scaleset = [[0.5, 0.0, 0.3, 0.1],
            [0.4, 0.15, 0.2, 0.1],
            [0.2, 0.4, 0.7, 0.8]]

twod = clusters.scaledown(scaleset, distance=euclidean)
clusters.draw2d(twod, labels, jpeg='abcd.jpg')
