#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import math

matrix = np.loadtxt('./output/mat.txt', usecols=range(1023))

times = [0.530031, 1.01834, 3.16735, 12.0174, 49.3414]
sizes = [1024, 2048, 4096, 8192, 16384]

"""
plt.matshow(matrix)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Poisson")
plt.savefig("./output/poisson2.png")
plt.show()
"""

#plt.semilogy(sizesLog, times, "*")
fig, ax = plt.subplots()

ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
ax.plot(sizes, times, "*")
plt.xlabel("Problem size ($n$)")
plt.ylabel("Time ($s$)")
plt.title("Timings for different problem sizes")
plt.savefig("./output/problemsize.png")
plt.show()
