#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import math

matrix = np.loadtxt('./output/mat.txt', usecols=range(1023))

timesPureMPI = [0.530031, 1.01834, 3.16735, 12.0174, 49.3414]
sizesPureMPI = [1024, 2048, 4096, 8192, 16384]

timesMaxOMP = [142.401, 80.2883, 57.0892, 46.5551, 37.4316, 37.7438, 37.4435, 37.6526]
threadsMaxOMP = [4, 8, 12, 16, 20, 24, 28, 32]

timesHybrid = [39.6883, 39.2322, 46.5551, 46.3064, 45.8982]
xcoordHybrid = [0, 1, 2, 3, 4]
labelsHybrid = ["(1 x 16, 1)", "(2 x 8, 1)", "(2 x 8, 2)", "(4 x 4, 2)", "(8 x 2, 2)"]



#  Plot for the solution of the Poisson problem
plt.matshow(matrix)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Poisson")
plt.savefig("./output/poisson2.png")
plt.show()


#plt.semilogy(sizesLog, times, "*")
fig, ax = plt.subplots()

"""
# Plot for timesPureMPI
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
ax.plot(sizesPureMPI, timesPureMPI, "*")
ax.plot(sizesPureMPI, timesPureMPI, "--", alpha=0.25)
plt.xlabel("Problem size ($n$)")
plt.ylabel("Time ($s$)")
plt.title("Timings for different problem sizes")
plt.savefig("./output/problemsize.png")
plt.show()
"""
"""
# Plot for timesMaxOMP
ax.plot(threadsMaxOMP, timesMaxOMP, "*",)
ax.plot(threadsMaxOMP, timesMaxOMP, "--",alpha=0.25)
plt.xlabel("Number of threads")
plt.ylabel("Time ($s$)")
plt.title("Timings for different numbers of threads")
plt.savefig("./output/timingsthread.png")
plt.show()
"""
"""
plt.xlabel("Configuration (<p> x <t>, <N>)")
plt.ylabel("Time ($s$)")
plt.title("Run time comparison with hybrid models")
plt.bar(xcoordHybrid, timesHybrid, width=0.8, tick_label=labelsHybrid)
plt.savefig("./output/hybridcomp.png")
plt.show()
"""
