import numpy as np
from matplotlib import pyplot as plt
from geneticalgorithm import geneticalgorithm as ga


m = 20
n = 100
k = 10
x = np.arange(0.0, k, 0.1)
a = np.linspace(10, n, n)
t = np.linspace(0, n, n)


def y(x):
    return 1+np.exp(-t/m)


def f(x):
    return -0.01*a+2.1


def g(x):
    return 2


def h(x):
    return g(x)+f(x)-y(x)


plt.plot(x, np.full(x.shape, g(x)), color='red')
plt.plot(a, f(x), color='red')
plt.plot(t, y(x), color='blue')
plt.show()
plt.plot(t, h(x), color='green')
plt.show()


def q(X):
    dim = len(X)

    OF = 0
    for i in range(0, dim):
        #OF += (1+np.exp(-X[i]/m)+0.01*X[i]-2.1-2)
        OF += (2-0.01*X[i]+2.1-1-np.exp(-X[i]/m))
    return -OF


varbound = np.array([[0, 98]])

algorithm_param = {'max_num_iteration': 250,
                   'population_size': 200,
                   'mutation_probability': 0.05,
                   'elit_ratio': 0.05,
                   'crossover_probability': 0.6,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}


model = ga(function=q, dimension=1, variable_type='real',
           variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()

# convergence = model.report
# #solution = model.ouput_dict
