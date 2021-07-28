import matplotlib.pyplot as plt
import numpy as np

# This script assumes you're running from the linfa-clustering root after
# running the example so the dataset and reachability npy files are in the 
# linfa-clustering root as well.

dataset = np.load("../dataset.npy")
reachability = np.load("../reachability.npy")

plot1 = plt.figure(1)
plt.scatter(dataset[:, 0], dataset[:, 1])

plot2 = plt.figure(2)
plt.plot(reachability)

plt.show()
