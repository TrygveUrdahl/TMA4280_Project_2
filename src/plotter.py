#!/usr/local/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('./output/mat.txt', usecols=range(1023))
#fig = plt.figure()
#ax = plt.axes(projection='3d')

plt.matshow(matrix)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Poisson")
plt.show()

#print(matrix)
