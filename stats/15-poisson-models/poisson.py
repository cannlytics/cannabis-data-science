"""
Poisson Distribution
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/4/2022
Updated: 3/4/2022
License: MIT License <https://opensource.org/licenses/MIT>

Original plot by: Skbkekas
License: https://creativecommons.org/licenses/by/3.0/deed.en
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 12)

# Define the data.
col = {
    1 : 'orange',
    4: 'purple', 
    10: 'lightblue'
}
X = np.arange(0, 21)


plt.clf()
plt.figure(figsize=(4, 3.2))

A = []
for _lambda in 1, 4, 10:

    # Evaluate the Poisson distribution
    P = -_lambda + X * np.log(_lambda) - sp.gammaln(X + 1)
    P = np.exp(P)

    # Plot the line.
    plt.plot(X, P, '-', color='grey', label='_nolegend_')

    # Plot the point.
    a = plt.plot(X, P, 'o', color=col[_lambda], markeredgecolor='k', markeredgewidth=0.5)
    A.append(a)

# Add labels.
plt.xlabel("$k$")
plt.ylabel(r"$P(x=k)$")
plt.ylim([0, 0.4])
plt.yticks(np.arange(0, 0.42, 0.05))
plt.tick_params(direction='in', top=True, right=True)
plt.legend((r"$\lambda=1$", r"$\lambda=4$", r"$\lambda=10$"),
                numpoints=1, handlelength=0.75, handletextpad=0.5,\
                loc="upper right", frameon=False)
plt.xlim(-1,21)

# Save the figure.
plt.savefig(
    'poisson_pmf.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False
)


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')
sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
plt.legend(loc='best')
plt.xlabel("$y$")
plt.ylabel(r"$P(y|\theta)$")
plt.savefig(
    'poisson_normal_binomial.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False
)
plt.show()
