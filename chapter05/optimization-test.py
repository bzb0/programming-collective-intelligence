import optimization

optimization.loadFlights()

schedule = [1, 4, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]
print('Schedule cost for manual selection: {:d}'.format(optimization.schedulecost(schedule)))
optimization.printschedule(schedule)

print('*******************************************************')

domain = [(0, 8)] * (len(optimization.people) * 2)
randschedule = optimization.randomoptimize(domain, optimization.schedulecost)
print('Schedule cost for random optimization: {:d}'.format(optimization.schedulecost(randschedule)))
optimization.printschedule(randschedule)

print('*******************************************************')

hillclimbschedule = optimization.hillclimb(domain, optimization.schedulecost)
print('Schedule cost for hill climb: {:d}'.format(optimization.schedulecost(hillclimbschedule)))
optimization.printschedule(hillclimbschedule)

print('*******************************************************')

simannealschedule = optimization.annealingoptimize(domain, optimization.schedulecost)
print('Schedule cost for simulated annealing: {:d}'.format(optimization.schedulecost(simannealschedule)))
optimization.printschedule(simannealschedule)

print('*******************************************************')

geneticschedule = optimization.geneticoptimize(domain, optimization.schedulecost)
print('Schedule cost for genetic algorithm: {:d}'.format(optimization.schedulecost(geneticschedule)))
optimization.printschedule(geneticschedule)
