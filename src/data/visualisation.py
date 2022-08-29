import matplotlib.pyplot as plt
import numpy as np

from dataloader1 import Dataset

dataset = Dataset("C:/Users/Bruger/Documents/1Speciale/Speciale/data/open/WESAD/")

# A single time-series for BVP
series = dataset["S2"]["wrist"]["EDA"]
labels = dataset["S2"]["label"]

# visualise
n = 64 * 100 # number of seconds to visualise (here 100s)
plt.plot(range(n),series[:n])
plt.grid()
plt.xlabel("data-points")
plt.ylabel("BVP-values")
plt.title("BVP-series")
plt.show()

# distribution
n, bins, patches = plt.hist(x=series, bins="auto", color="#0504aa",alpha=0.7, rwidth=0.85)
plt.grid(axis="y", alpha=0.75)
plt.xlabel("values")
plt.ylabel("frequency")
plt.title("Histogram")
maxfreq = n.max()
# set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
