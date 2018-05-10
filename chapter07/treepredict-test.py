import treepredict

# my_data = [line.split('\t') for line in file('decision_tree_example.txt')]
my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]

set1, set2 = treepredict.divideset(my_data, 2, 'yes')

print("Set-True: " + str(set1))
print("Set-False: " + str(set2))

treepredict.entropy(set1)
treepredict.giniimpurity(set1)

print("Gini Impurity %4.3f", treepredict.giniimpurity(my_data))
print("Entropy %4.3f", treepredict.entropy(my_data))

tree = treepredict.buildtree(my_data)
treepredict.printtree(tree)
treepredict.drawtree(tree, jpeg='treeview.jpg')

print("\nTree classify: {:s}".format(str(treepredict.classify(['(direct)', 'USA', 'yes', 5], tree))))

treepredict.prune(tree, 1.0)
treepredict.printtree(tree)

print("\nClassifying with incomplete data")
print("Tree classify for incomplete data: {:s}".format(str(treepredict.mdclassify(['google', None, 'yes', None], tree))))
print("Tree classify for incomplete data: {:s}".format(str(treepredict.mdclassify(['google', 'France', None, None], tree))))
