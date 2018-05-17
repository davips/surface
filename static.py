import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
import numpy as np


# object for keeping instance data
class INST:
    pass


inst = INST()
inst.t = 1  # time for probing
inst.s = 1  # traveling speed
inst.T = 40  # time limit for a route
inst.x0 = 0  # depot coordinates
inst.y0 = 0  # depot coordinates


# auxiliary function: euclidean distance
def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# find max variance in a grid
# retorna lista ordenada por variância
def max_var(gpr, n):
    # prepare data
    x = np.arange(0, 1.01, 1. / n)
    y = np.arange(0, 1.01, 1. / n)
    X = [(x[i], y[j]) for i in range(n + 1) for j in range(n + 1)]
    z, z_std = gpr.predict(X, return_std=True)
    points = [(x, v) for (v, x) in sorted(zip(zip(z_std, z), X))]
    # print(z_std)
    # print(X)
    # for i in range(len(X)):
    #     print("\t{}\t{}\t{}".format(i,X[i],z_std[i]))
    # print(points)
    return points

# def max_expected_var_red(n, xys, zs):
#     # build model
#     X = list(xyz)  # work with local copies
#     z = list(zs)
#     from tsp import solve_tsp, sequence  # exact
#     from tsp import multistart_localsearch  # heuristic
#     kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
#     gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
#     gpr.fit(X, z)
#


# required function: route planning
def planner(X, z, f, dynam=False):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - f: useless in static version
    """
    X = list(X)  # work with local copies
    z = list(z)
    from tsp import solve_tsp, sequence  # exact
    from tsp import multistart_localsearch  # heuristic
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # # plot preliminary GP
    # from functions import plot
    # from functions import f1
    # plot(f1,100)
    # # end of plot
    # # # plot posteriori GP
    # GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # GPR.fit(X, z)
    # def GP(x_,y_):
    #     return GPR.predict([(x_,y_)])[0]
    # plot(GP,100)
    # # # end of plot

    x0, y0 = (inst.x0, inst.y0)  # initial position (depot)
    pos = [(x0, y0)]
    N = 1
    c = {}
    while True:
        # attempt adding Nth "city" to the tour

        # retorna lista ordenada por variância
        points = max_var(gpr, n=100)  # n x n grid !!!!!

        # pega ponto com maior variância
        (x, y), (z_new_std, z_new) = points.pop()

        # descarta previamente já selecionados, passando para a próxima maior variância
        while (x, y) in pos:
            (x, y), (z_new_std, z_new) = points.pop()

        # estende matriz de distâncias para ambos solvers
        for i in range(N):
            c[i, N] = dist(*pos[i], x, y)
            c[N, i] = c[i, N]

        # evita TSP em menos de 3 'cidades', falta checar limite de tempo
        if N < 3:
            N += 1
            pos.append((x, y))
            continue  # !!!!!

        sol_, cost = multistart_localsearch(100, N + 1, c, cutoff=inst.T - (N) * inst.t)  # heuristic
        if cost <= inst.T - (N) * inst.t:
            print("heuristic solution")
            idx = sol_.index(0)
            sol = sol_[idx:] + sol_[:idx]  # heuristic
            # print(obj + (N)*inst.t, "TSP solution:", obj, N, inst.T, sol)
            # print("appending", (x,y), z_new_std, z_new, "orient.len:", obj + (N)*inst.t)
            N += 1
            assert (x, y) not in pos
            pos.append((x, y))
            X.append((x, y))
            if (dynam):
                z.append(f(x, y))  # !!!!! with PROBING
            else:
                z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
        else:
            # attempt exact solution:
            print("attempting exact solution")
            cost, edges = solve_tsp(range(N + 1), c)  # exact
            if cost <= inst.T - (N) * inst.t:
                sol = sequence(range(N + 1), edges)  # exact
                # print(obj + (N) * inst.t, "TSP EXACT:", obj, N, inst.T, sol)
                # print("appending", (x, y), z_new_std, z_new, "orient.len:", obj + (N) * inst.t)
                N += 1
                pos.append((x, y))
                X.append((x, y))
                if (dynam):
                    z.append(f(x, y))  # !!!!! with PROBING
                else:
                    z.append(z_new)  # !!!!! with average of the prediction as an estimate
                gpr.fit(X, z)
                print("found; continue")
                continue
            print("heuristic and exact solution exceeds time limit")

            # print("testing next interesting points")

            print("break")
            break

    print(cost + (N) * inst.t, "TSP solution:", cost, N, inst.T, sol)
    return [pos[i] for i in sol[1:]]


# required function: route planning with probing
def explorer(X, z, plan):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - plan: list of coordinates of initial probing plan [(x1,y1), ...., (xP,yP)]
                this plan may be changed; the only important movement is (x1,y1)
    """
    X = list(X)  # work with local copies
    z = list(z)
    from tsp import solve_tsp, sequence
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

    x0, y0 = (inst.x0, inst.y0)  # initial position (depot)
    x1, y1 = plan[0]
    pos = [(x0, y0), (x1, y1)]
    N = 2
    c = {}
    c[0, 1] = dist(*pos[0], *pos[1])
    while True:
        # attempt adding Nth "city" to the tour
        points = max_var(gpr, n=100)  # n x n grid !!!!!
        (x, y), (z_new_std, z_new) = points.pop()
        for i in range(N):
            c[i, N] = dist(*pos[i], *(x, y))

        if N < 3:
            N += 1
            pos.append((x, y))
            continue  # !!!!!
        obj, edges = solve_tsp(range(N + 1), c)
        if obj <= inst.T - (N) * inst.t:
            sol = sequence(range(N + 1), edges)
            print("TSP solution:", obj, N, inst.T, sol)
            print("appending", (x, y), z_new_std, z_new)
            N += 1
            pos.append((x, y))
            X.append((x, y))
            z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
        else:
            # maybe some other elements of 'points' should be attempted here
            break

    return [pos[i] for i in sol[1:]]


def estimator(X, z, mesh):
    """estimator: treina em X, z e retorna valores preditos para os pontos no mesh
        evaluate z at points [(x1,y1), ...., (xK,yK)] based on M=n+N known points:
        - X: list of coordinates [(x1,y1), ...., (xM,yM)]
        - z: list of evaluations of "true" function [z1, ..., zM]
    """
    # from tools import trainGP
    # gpr = trainGP(X, z)

    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    gpr.fit(X, z)

    z = []
    for (x_, y_) in mesh:
        val = gpr.predict([(x_, y_)])
        z.append(val)

    return z
