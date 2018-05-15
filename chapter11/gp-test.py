import gp

exampletree = gp.exampletree()
print("Example function result for values [2,3]: {:d}".format(exampletree.evaluate([2, 3])))
print("Example function result for values [5,3]: {:d}".format(exampletree.evaluate([5, 3])))

mytree = gp.mytree()
print("My tree result for values [101,5]: {:d}".format(mytree.evaluate([101, 5])))
print("My tree result for values [99,5]: {:d}".format(mytree.evaluate([99, 5])))

print("Example tree:")
exampletree.display()

print("My tree:")
mytree.display()

random1 = gp.makerandomtree(2)
print("Random tree1 output for input[7,1]: {:d}".format(random1.evaluate([7, 1])))
print("Random tree1 output for input[2,4]: {:d}".format(random1.evaluate([2, 4])))
print("Random tree1:")
random1.display()

random2 = gp.makerandomtree(2)
print("Random tree2 output for input[5,3]: {:d}".format(random2.evaluate([5, 3])))
print("Random tree2 output for input[5,20]: {:d}".format(random2.evaluate([5, 20])))
print("Random tree2:")
random2.display()

hiddenset = gp.buildhiddenset()
print("Hidden set results:")
for row in hiddenset:
    print("{:2d}, {:2d} => {:4d}".format(row[0], row[1], row[2]))

print("Score function output for tree1: {:8d}".format(gp.scorefunction(random1, hiddenset)))
print("Score function output for tree2: {:8d}".format(gp.scorefunction(random2, hiddenset)))

muttree = gp.mutate(random2, 2)
print("Mutated tree:")
muttree.display()
print("Score function output for mutated tree (source: tree2): {:8d}".format(gp.scorefunction(muttree, hiddenset)))

cross = gp.crossover(random1, random2)
print("Crossover tree:")
cross.display()
print("Score function output for crossover tree (random1, random2): {:8d}".format(gp.scorefunction(cross, hiddenset)))

rf = gp.getrankfunction(gp.buildhiddenset())
gp.evolve(2, 500, rf, mutationrate=0.2, breedingrate=0.1, pexp=0.7, pnew=0.1)

player1 = gp.makerandomtree(5)
player2 = gp.makerandomtree(5)
print("After 50 tries for random trees Player {:d} won".format(gp.gridgame([player1, player2])))

winner = gp.evolve(5, 100, gp.tournament, maxgen=50)

gp.gridgame([winner, gp.humanplayer()])
