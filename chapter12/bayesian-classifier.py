import docclass

print("Words for input {:s}".format(str(docclass.getwords('python is a dynamic language'))))

cl = docclass.naivebayes(docclass.getwords)
cl.setdb('test.db')

cl.train('pythons are constrictors', 'snake')
cl.train('python has dynamic types', 'language')
cl.train('python was developed as a scripting language', 'language')

print("Input 'dynamic programming' classified as {:s}".format(cl.classify('dynamic programming')))
print("Input 'boa constrictors' classified as {:s}".format(cl.classify('boa constrictors')))
