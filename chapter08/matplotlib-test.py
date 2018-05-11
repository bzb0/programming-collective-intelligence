import matplotlib.pyplot as plt
import numpy as np

a = [1, 2, 3, 4]
b = [4, 2, 3, 1]

plt.plot(a, b)  # Plot some data on the axes.
plt.show()

t1 = np.arange(0.0, 10.0, 0.1)
plt.plot(t1, np.sin(t1))
plt.show()
