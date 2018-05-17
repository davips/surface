from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
import math
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Esse import evita "ValueError: Unknown projection '3d'"
from functions import f5, f10
from tsp import solve_tsp, sequence  # exact
from tsp import multistart_localsearch  # heuristic
import time

exact = True
n = 10
SEED = 142


def kernel_selection(xys, zs):
    # define limits of the hyperparameter space
    # bounds = [(0.00001, 0.0001), (0.0001, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 100000)]
    # nu_bounds = [0.1, 0.5, 1, 1.5, 2, 2.5, 5, 20]
    bounds = [(0.00001, 0.001), (0.001, 0.1), (0.1, 10), (10, 1000), (1000, 100000)]
    nu_bounds = [0.5, 1.5, 2.5]

    # generate list of kernels to assess
    quads = [RationalQuadratic(length_scale_bounds=(a, b), alpha_bounds=(c, d)) for a, b in bounds for c, d in bounds]
    rbfs = [RBF(length_scale_bounds=(a, b)) for a, b in bounds]
    mtns = [Matern(length_scale_bounds=(a, b), nu=c) for a, b in bounds for c in nu_bounds]
    kernels = quads + rbfs + mtns
    # ExpSineSquared(),    # kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))

    # find best kernel on k-fold CV
    min_error = 99999
    for kernel in kernels[:1]:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
        err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
        # print((type(kernel).__name__[:12] + '\t:\t' + str(err)).expandtabs(13))
        if err < min_error:
            min_error = err
            min_error_kernel = kernel

    # fit using all training data and best kernel
    gpr = GaussianProcessRegressor(kernel=min_error_kernel, n_restarts_optimizer=5, copy_X_train=True)
    gpr.fit(xys, zs)
    return gpr


def train_data(seed=SEED):
    return data(f5, math.ceil(n / 3), seed + 1)


def test_data(seed=SEED):
    return data(f5, n, seed + 2)


def data(f, n2, seed=SEED):
    random.seed(seed)
    xys, zs = [], []
    for x in [i / n2 for i in range(1, n2)]:
        for y in [i / n2 for i in range(1, n2)]:
            x = random.random()
            y = random.random()
            xys.append((x, y))
            zs.append(f(x, y))
            # print("{}\t&{}\t&{}\\\\".format(x, y, zs[-1]))
    return xys, zs


def gp(xys, zs):
    from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic
    from sklearn.gaussian_process import GaussianProcessRegressor
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, copy_X_train=True)
    gpr.fit(xys, zs)
    return gpr


def max_var(g):
    xs = np.arange(0.0, 1.01, 1. / n)
    ys = np.arange(0.0, 1.01, 1. / n)
    xys = [(xs[i], ys[j]) for i in range(n + 1) for j in range(n + 1)]
    zs, stds = g.predict(xys, return_std=True)
    stdmax, zmax, xymax = max(zip(stds, zs, xys))
    return xymax, zmax, stdmax


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def complete_cost(cost, n):
    # each probing has cost 1
    return cost + n


def plan_tour(xys, budget):
    n = len(xys)
    pos = xys
    c = {}
    for i in range(n):
        for j in range(n):  # non optimized distance calculations!
            c[i, j] = dist(*pos[i], *pos[j])
            c[j, i] = c[i, j]

    if n > 2:
        # heuristic
        sol_, cost = multistart_localsearch(100, n, c, cutoff=budget - n)  # inst.T - (N) * inst.t)
        idx = sol_.index(0)
        sol = sol_[idx:] + sol_[:idx]
        cost = complete_cost(cost, n)

        # exact
        if cost > budget:
            if exact:
                cost, edges = solve_tsp(range(n), c)
                cost = complete_cost(cost, n)
                sol = sequence(range(n), edges)
            else:
                print('NOT trying exact solution')
    else:
        cost = dist(*pos[0], *pos[1])
        cost = complete_cost(cost, n)
        sol = [0, 1]

    # if cost <= budget:
    #     print('>>>>>>>>>>>>>>>>> Feasible= total cost\t', fmt(cost), '\tis less than\t', fmt(budget))
    # else:
    #     print("Unfeasible.")

    return sol, cost <= budget, cost


def evalu_max(g, xys, zs):
    predzs = g.predict(xys, return_std=False)
    return max([abs(a - b) for a, b in zip(zs, predzs)])


def evalu_sum(g, xys, zs):
    predzs = g.predict(xys, return_std=False)
    # print([abs(a - b) for a, b in zip(zs, predzs)])
    return sum([abs(a - b) for a, b in zip(zs, predzs)])


def evalu_var(g, xys, zs):
    predzs, stds = g.predict(xys, return_std=True)
    return sum(stds)


def fmt(txt):
    return "{:6.6f}".format(txt)


def show_path(path, label=''):
    for a, b in path:
        print(fmt(a), '\t', fmt(b), '\t\t=path\t', label)


def p(plt, fig, f, n, zmin, zmax, filename=None):
    import math
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np

    # prepare data
    X = np.arange(0, 1.01, 1. / n)
    Y = np.arange(0, 1.01, 1. / n)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros((n + 1, n + 1))
    W = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            Z[i, j] = f(X[i, j], Y[i, j])

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #    ax.set_zlabel('z(x,y)')
    ax.set_zlim(zmin, zmax)
    pl = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)
    # plt.show()
    fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    time.sleep(0.2)
    pl.remove()
