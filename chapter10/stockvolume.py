import csv

import nmf
import numpy as np

tickers = ['AAPL', 'BIIB', 'BP', 'CL', 'CVX', 'EXPE', 'GE', 'GOOG', 'GS', 'JPM', 'MSFT', 'PG', 'XOM']

shortest = 300
prices = {}
dates = []

for ticker in tickers:
    with open('stockdata/' + ticker + '.csv', 'r') as tickerCsv:
        csvReader = csv.DictReader(tickerCsv, delimiter=',')
        # Extract the volume field from every line
        for row in csvReader:
            prices.setdefault(ticker, [])
            prices[ticker].append(float(row['Volume']))
            dates.append(row['Date'])

    if len(prices[ticker]) < shortest:
        shortest = len(prices[ticker])

l1 = [[prices[tickers[i]][j] for i in range(len(tickers))] for j in range(shortest)]

w, h = nmf.factorize(np.matrix(l1), pc=5)
print("h: {:s}, w: {:s}".format(str(h), str(w)))

# Loop over all the features
for i in range(np.shape(h)[0]):
    print("Feature {:d}".format(i))

    # Get the top stocks for this feature
    ol = [(h[i, j], tickers[j]) for j in range(np.shape(h)[1])]
    ol.sort()
    ol.reverse()
    for j in range(12):
        print("{:s}".format(str(ol[j])))
    print("\n")

    # Show the top dates for this feature
    porder = [(w[d, i], d) for d in range(300)]
    porder.sort()
    porder.reverse()

    print("{:s}\n".format(str([(p[0], dates[p[1]]) for p in porder[0:3]])))
