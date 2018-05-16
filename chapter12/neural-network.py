import nn

online, pharmacy = 1, 2
spam, notspam = 1, 2

possible = [spam, notspam]

neuralnet = nn.searchnet('nntest.db')
neuralnet.maketables()

neuralnet.trainquery([online], possible, notspam)
neuralnet.trainquery([online, pharmacy], possible, spam)
neuralnet.trainquery([pharmacy], possible, notspam)

print("Input 'online pharmacy' classified as: {:s}'".format(str(neuralnet.getresult([online, pharmacy], possible))))
print("Input 'online' classified as: {:s}'".format(str(neuralnet.getresult([online], possible))))

neuralnet.trainquery([online], possible, notspam)
print("Input 'online' classified as: {:s}'".format(str(neuralnet.getresult([online], possible))))

neuralnet.trainquery([online], possible, notspam)
print("Input 'online' classified as: {:s}'".format(str(neuralnet.getresult([online], possible))))
