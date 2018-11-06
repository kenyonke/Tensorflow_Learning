# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:39:09 2018

@author: kenyon
"""
import numpy as np
from matplotlib import pyplot as plt

#generate data
x = np.random.normal(size=4)  # create four (4) data points
noise = np.random.normal(scale=0.5, size=4) # standard deviation of the noise: 0.5
m_true = 1.4  # true value for slope
c_true = -3.1  # true value for offset
y = m_true * x + c_true + noise #training data

m_star = 0.0  # guess (initial value) for slope
c_star = -5.0  # guess (initial value) for offset

#training process
def regression_contour_fit(ax, iterations, learn_rate, m_star, c_star, x, y):
    for i in range(iterations):

        # update offset
        c_grad = -2 * (y - m_star * x - c_star).sum()
        c_star = c_star - learn_rate * c_grad

        # update slope
        m_grad = -2 * (x * (y - m_star * x - c_star)).sum()
        m_star = m_star - learn_rate * m_grad

        # plot the current position
        ax.plot(m_star, c_star, 'g*', markersize=10)

# define a function for contour plot
def regression_contour(f, ax, m_vals, c_vals, E_grid):

    # contour plot
    hcont = ax.contour(m_vals, c_vals, E_grid, levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64])

    # contour labels
    plt.clabel(hcont, inline=1, fontsize=15)

    # axis labels
    ax.set_xlabel('$m$', fontsize=25)
    ax.set_ylabel('$c$', fontsize=25)
        
# contour plot
f, ax = plt.subplots(figsize=(8, 6))

# compute the error function at each combination of c and m
m_vals = np.linspace(m_true-3, m_true+3, 100) 
c_vals = np.linspace(c_true-3, c_true+3, 100)
m_grid, c_grid = np.meshgrid(m_vals, c_vals)
E_grid = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        E_grid[i, j] = ((y - m_grid[i, j] * x - c_grid[i, j])**2).sum()
regression_contour(f, ax, m_vals, c_vals, E_grid)
ax.set_title('gradient descent', fontsize=25)

# plot the initial position
m_star = 0.0
c_star = -5.0
ax.plot(m_star, c_star, 'g*', markersize=10)

# plot the updated positions
iterations = 100  # number of iterations
learn_rate = 0.01  # learning rate
regression_contour_fit(ax, iterations, learn_rate, m_star, c_star, x, y)