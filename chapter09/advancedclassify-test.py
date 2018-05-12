import array

from sklearn import svm

import advancedclassify


def classToString(clazz):
    if isinstance(clazz, array.__class__):
        return "match" if clazz[0] == 1 else "no match"
    else:
        return "match" if clazz == 1 else "no match"


agesonly = advancedclassify.loadmatch('agesonly.csv', allnum=False)
matchmaker = advancedclassify.loadmatch('matchmaker.csv')

advancedclassify.plotagematches(agesonly)

avgs = advancedclassify.lineartrain(agesonly)
print("Age averages: {:s}".format(str(avgs)))

print("Classifying ages (30, 30) as: {:s}".format(classToString(advancedclassify.dpclassify([30, 30], avgs))))
print("Classifying ages (30, 25) as: {:s}".format(classToString(advancedclassify.dpclassify([30, 25], avgs))))
print("Classifying ages (25, 40) as: {:s}".format(classToString(advancedclassify.dpclassify([25, 40], avgs))))
print("Classifying ages (48, 20) as: {:s}".format(classToString(advancedclassify.dpclassify([48, 20], avgs))))

numericalset = advancedclassify.loadnumerical()
scaledset, scalef = advancedclassify.scaledata(numericalset)
avgs = advancedclassify.lineartrain(scaledset)

print("Match maker data[0]: {:s}".format(str(numericalset[0].data)))
print("Classifying match maker data[0] as: {:s}".format(classToString(advancedclassify.dpclassify(scalef(numericalset[0].data), avgs))))

print("Match maker data[11]: {:s}".format(str(numericalset[11].data)))
print("Classifying match maker data[11] as: {:s}".format(classToString(advancedclassify.dpclassify(scalef(numericalset[11].data), avgs))))

offset = advancedclassify.getoffset(agesonly, gamma=20)

print("Classifying ages non-linear (30, 30) as: {:s}".format(classToString(advancedclassify.nlclassify([30, 30], agesonly, offset))))
print("Classifying ages non-linear (30, 25) as: {:s}".format(classToString(advancedclassify.nlclassify([30, 25], agesonly, offset))))
print("Classifying ages non-linear (25, 40) as: {:s}".format(classToString(advancedclassify.nlclassify([25, 40], agesonly, offset))))
print("Classifying ages non-linear (48, 20) as: {:s}".format(classToString(advancedclassify.nlclassify([48, 20], agesonly, offset))))

ssoffset = advancedclassify.getoffset(scaledset)

print("Match maker data[0]: {:s}".format(str(numericalset[0].match)))
print("Classifying match maker data[0] as: {:s}".format(classToString(advancedclassify.nlclassify(scalef(numericalset[0].data), scaledset, ssoffset))))

print("Match maker data[1]: {:s}".format(str(numericalset[1].match)))
print("Classifying match maker data[1] as: {:s}".format(classToString(advancedclassify.nlclassify(scalef(numericalset[1].data), scaledset, ssoffset))))

newrow = [28.0, -1, -1, 26.0, -1, 1, 2, 0.8]  # Man doesn't want children, woman does
print("Classifying an entry where a man doesn't want children and woman does as: {:s}"
      .format(classToString(advancedclassify.nlclassify(scalef(newrow), scaledset, ssoffset))))

newrow = [28.0, -1, 1, 26.0, -1, 1, 2, 0.8]  # Both want children
print("Classifying an entry where both want children as: {:s}"
      .format(classToString(advancedclassify.nlclassify(scalef(newrow), scaledset, ssoffset))))

answers, inputs = [r.match for r in scaledset], [r.data for r in scaledset]

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(inputs, answers)

newrow = [28.0, -1, -1, 26.0, -1, 1, 2, 0.8]  # Man doesn't want children, woman does
print("SVM classify for an entry where a man doesn't want children and woman does as: {:s}".format(classToString(rbf_svc.predict([scalef(newrow)]))))

newrow = [28.0, -1, 1, 26.0, -1, 1, 2, 0.8]  # Both want children
print("SVM classify for an entry where both want children as: {:s}".format(classToString(rbf_svc.predict([scalef(newrow)]))))
