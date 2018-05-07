import math
import random
import time


def getminutes(t):
    x = time.strptime(t, '%H:%M')
    return x[3] * 60 + x[4]


people = [('Seymour', 'BOS'), ('Franny', 'DAL'), ('Zooey', 'CAK'), ('Walt', 'MIA'), ('Buddy', 'ORD'), ('Les', 'OMA')]
# LaGuardia airport in New York
destination = 'LGA'

flights = {}

def loadFlights():
    file = open('schedule.txt', 'r')
    for line in file.readlines():
        origin, dest, depart, arrive, price = line.strip().split(',')
        flights.setdefault((origin, dest), [])
        # Add details to the list of possible flights
        flights[(origin, dest)].append((depart, arrive, int(price)))


# r is given in the format [outbound_person1, return_person2, ..., outbound_personN, return_personN]
def printschedule(r):
    for d in range(int(len(r) / 2)):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][r[d]]
        ret = flights[(destination, origin)][r[d + 1]]
        print('{:10s}{:10s} {:5s}-{:5s} ${:3d} {:5s}-{:5s} ${:3d}'.format(name, origin, out[0], out[1], out[2], ret[0], ret[1], ret[2]))


# Cost function, for solving the flight scheduling problem
def schedulecost(schedule):
    totalprice = 0
    latestarrivalInMinutes = 0
    earliestdepartInMinutes = 24 * 60

    for d in range(int(len(schedule) / 2)):
        # Get the inbound and outbound flights
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(schedule[d])]
        returnf = flights[(destination, origin)][int(schedule[d + 1])]
        # Total price is the price of all outbound and return flights
        totalprice += outbound[2]
        totalprice += returnf[2]
        # Track the latest arrival and earliest departure
        if latestarrivalInMinutes < getminutes(outbound[1]):
            latestarrivalInMinutes = getminutes(outbound[1])
        if earliestdepartInMinutes > getminutes(returnf[0]):
            earliestdepartInMinutes = getminutes(returnf[0])

    # Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    totalwait = 0
    for d in range(int(len(schedule) / 2)):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(schedule[d])]
        returnf = flights[(destination, origin)][int(schedule[d + 1])]
        totalwait += latestarrivalInMinutes - getminutes(outbound[1])
        totalwait += getminutes(returnf[0]) - earliestdepartInMinutes

    # Does this solution require an extra day of car rental? That'll be $50!
    if latestarrivalInMinutes > earliestdepartInMinutes:
        totalprice += 50

    return totalprice + totalwait


def randomoptimize(domain, costfunction):
    bestcost = 999999999
    bestsolution = None

    for i in range(1000):
        # Create a random solution
        solution = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        # Get the cost
        cost = costfunction(solution)
        # Compare it to the best one so far
        if cost < bestcost:
            bestcost = cost
            bestsolution = solution

    return bestsolution


def hillclimb(domain, costf):
    # Create a random solution
    solution = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

    # Main loop
    while 1:
        # Create list of neighboring solutions
        neighbors = []
        for j in range(len(domain)):
            # One away in each direction
            if solution[j] > domain[j][0]:
                # Concatenate these three lists (before j, value+1, after j): schedule[0:j] + [current_value + 1] + schedule[j+1:]
                neighbors.append(solution[0:j] + [solution[j] - 1] + solution[j + 1:])
            if solution[j] < domain[j][1]:
                neighbors.append(solution[0:j] + [solution[j] + 1] + solution[j + 1:])

        # See what the best solution amongst the neighbors is
        current = costf(solution)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost < best:
                best = cost
                solution = neighbors[j]

        # If there's no improvement, then we've reached the top
        if best == current:
            break

    return solution


def annealingoptimize(domain, costf, temperature=100000.0, cool=0.95, step=1):
    # Initialize the values randomly
    vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

    while temperature > 0.1:
        # Choose one of the indices
        i = random.randint(0, len(domain) - 1)
        # Choose a direction to change it
        dir = random.randint(-step, step)
        # Create a new list with one of the values changed
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        # Calculate the current cost and the new cost
        currcost = costf(vec)
        newcost = costf(vecb)
        # We calculate the probability of a higher-cost solution being accepted
        p = pow(math.e, (-newcost - currcost) / temperature)

        # Is it better, or does it make the probability cutoff?
        if (newcost < currcost or random.random() < p):
            vec = vecb

        # Decrease the temperature
        temperature *= cool

    return vec

def geneticoptimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    """
    Finds a solution for a giving domain with the genetic algorithm.
    :param domain:
    :param costf: The cost function.
    :param popsize: The size of the population.
    :param step:
    :param mutprob: The probability that a new member of the population will be a mutation rather than a crossover.
    :param elite: The fraction of the population that are considered good solutions and are allowed to pass into the next generation.
    :param maxiter: The number of generations to run.
    :return:
    """

    # Mutation Operation
    def mutate(vec):
        #  We loop until we have found a liable item for change (otherwise we can cause an overflow/underflow/no-change)
        while True:
            i = random.randint(0, len(domain) - 1)
            if domain[i][0] <= (vec[i] - step) and (vec[i] + step) <= domain[i][1]:
                if vec[i] > domain[i][0]:
                    return vec[0:i] + [vec[i] - step] + vec[i + 1:]
                if vec[i] < domain[i][1]:
                    return vec[0:i] + [vec[i] + step] + vec[i + 1:]

    # Crossover Operation
    def crossover(r1, r2):
        i = random.randint(1, len(domain) - 2)
        return r1[0:i] + r2[i:]

    # Build the initial population
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # How many winners from each generation?
    topelite = int(elite * popsize)

    # Main loop
    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s, v) in scores]

        # Start with the pure winners
        pop = ranked[0:topelite]

        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random() < mutprob:
                # Mutation
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            else:
                # Crossover
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))

        # Print current best score
        # print scores[0][0]

    return scores[0][1]
