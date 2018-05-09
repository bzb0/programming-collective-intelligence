import docclass

cl = docclass.classifier(docclass.getwords)
cl.train('the quick brown fox jumps over the lazy dog', 'good')
cl.train('make quick money in the online casino', 'bad')
print("How many times 'quick' is found in class 'good': {:d}".format(cl.fcount('quick', 'good')))
print("How many times 'quick' is found in class 'bad': {:d}".format(cl.fcount('quick', 'bad')))

docclass.sampletrain(cl)
print("Probability of feature 'quick' in class 'good': {:5.4f}".format(cl.featureprob('quick', 'good')))
print("Weighted probability for 'money' in class 'good': {:5.4f}".format(cl.weightedprob('money', 'good', cl.featureprob)))

naivecl = docclass.naivebayes(docclass.getwords)
docclass.sampletrain(naivecl)

print("Naive Bayes probability for 'quick rabbit' in class 'good': {:5.4f}".format(naivecl.bayesprob('quick rabbit', 'good')))
print("Naive Bayes probability for 'quick rabbit' in class 'bad': {:5.4f}".format(naivecl.bayesprob('quick rabbit', 'bad')))

print("Classify (naive bayes) 'quick rabbit': {:s}".format(naivecl.classify('quick rabbit', default='unknown')))
print("Classify (naive bayes) 'quick money': {:s}".format(naivecl.classify('quick money', default='unknown')))
naivecl.setthreshold('bad', 3.0)
print("Classify (naive bayes) 'quick money': {:s}".format(naivecl.classify('quick money', default='unknown')))

for i in range(10):
    docclass.sampletrain(naivecl)
print("Classify (naive bayes) 'quick money': {:s}".format(naivecl.classify('quick money', default='unknown')))

fishercl = docclass.fisherclassifier(docclass.getwords)
docclass.sampletrain(fishercl)
print("Fisher Method probability for 'quick' in class 'good': {:5.4f}".format(fishercl.fisherprob('quick', 'good')))
print("Fisher Method probability for 'money' in class 'bad': {:5.4f}".format(fishercl.fisherprob('money', 'bad')))

print("Classify (fisher method) 'quick rabbit': {:s}".format(fishercl.classify('quick rabbit', default='unknown')))
print("Classify (fisher method) 'quick money': {:s}".format(fishercl.classify('quick money', default='unknown')))

fishercl.setminimum('bad', 0.8)
print("Classify (fisher method) 'quick rabbit': {:s}".format(fishercl.classify('quick rabbit', default='unknown')))
fishercl.setminimum('good', 0.4)
print("Classify (fisher method) 'quick money': {:s}".format(fishercl.classify('quick money', default='unknown')))

fishercl = docclass.fisherclassifier(docclass.getwords)
fishercl.setdb('fisherclassifier.db')
# docclass.sampletrain(fishercl)

print("Classify (fisher method) 'quick rabbit': {:s}".format(fishercl.classify('quick rabbit', default='unknown')))
print("Classify (fisher method) 'quick money': {:s}".format(fishercl.classify('quick money', default='unknown')))

fishercl.setminimum('bad', 0.8)
print("Classify (fisher method) 'quick rabbit': {:s}".format(fishercl.classify('quick rabbit', default='unknown')))
fishercl.setminimum('good', 0.4)
print("Classify (fisher method) 'quick money': {:s}".format(fishercl.classify('quick money', default='unknown')))
