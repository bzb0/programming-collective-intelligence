import treepredict

fruit = [[4, 'red', 'apple'],
         [4, 'green', 'apple'],
         [1, 'red', 'cherry'],
         [1, 'green', 'grape'],
         [5, 'red', 'apple']]

tree = treepredict.buildtree(fruit)

print("Classified fruit (2,red) as {:s}".format(str(treepredict.classify([2, 'red'], tree))))
print("Classified fruit (5,red) as {:s}".format(str(treepredict.classify([5, 'red'], tree))))
print("Classified fruit (1,green) as {:s}".format(str(treepredict.classify([1, 'green'], tree))))
print("Classified fruit (120,red) as {:s}".format(str(treepredict.classify([120, 'red'], tree))))

treepredict.classify([5, 'red'], tree)
treepredict.classify([1, 'green'], tree)
treepredict.classify([120, 'red'], tree)

treepredict.printtree(tree)
