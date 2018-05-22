from aux import *
from sys import argv
from numpy.random import normal
import matplotlib.pyplot as plt
import time

TSxy, TSz = test_data()

# whether it is dynamic (or static)
dynamic = argv[1] == 'dyn'
if dynamic: print("Dynamic mode!")

# generate list P with points from previous probing
Pxy, Pz = train_data()
depot = -0.0000001, -0.0000001

# create empty list N for new points
Nxy, Nz = [], []

# prepare plotter
plt.ion()
fig = plt.figure(num='surface')
minvar = 9999999

trip = Trip(depot, Pxy, Pz)

while True:
    while trip.isfeasible():

        # select best kernel and run GP on list P ++ N
        trip.refit(Nxy, Nz)

        # add point with highest variance to the list N (with simulated probing as z value)
        trip.addmaxvar()

        # define a tour departing from depot
        trip.plan_tour()

        if feasible:
            # store last succesful solution
            tour = tour_

            # plot path or surface
            plot_path(plt, fig, points, tour_)
            # p(plt, fig, f5, 30)
            # p(plt, fig, lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
            # p(plt, fig, lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)

            # evaluate (error sum, error max or var sum)
            var = evalu_var(g, TSxy)
            err = evalu_sum(g, TSxy, TSz)
            ast = ''
            if var < minvar:
                minvar = var
                ast = '*'

            print((type(g.kernel).__name__[:12]).expandtabs(13), '\ttour length=\t', len(tour_), '\tvar=\t' + fmt(var) + ast + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))
            # show_path([([depot] + Nxy)[i] for i in tour], '\tvar=\t' + fmt(var) + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))
        else:
            # drop last added, unfeasible, point
            Nxy.pop()
            Nz.pop()
            points.pop()

    # add depot to the beginning to allow triangulation (and to correct indexes to match tour indexes)
    Nxy = [depot] + Nxy
    Nz = [0] + Nz

    while not feasible:
        # modify almost all previous points, distorting them a bit at a time
        for ida, idb, idc in zip(tour, tour[1:], tour[2:]):
            (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
            m, n = (a + e) / 2, (b + f) / 2

            # # no distortion
            # x, y = c, d

            # random distortion
            s = 0.01 * (dist(a, b, c, d) + dist(c, d, e, f)) / 2
            x, y = c + normal(scale=s), d + normal(scale=s)

            # #   distortion towards median line = shortening the path
            # # offset = 0.1 * (dist(a, b, c, d) + dist(c, d, e, f) - dist(a, b, e, f))
            # p = 0.01
            # x, y = c + p * (m - c), d + p * (n - d)

            Nxy[idb] = x, y

        # check feasibility
        points = Nxy
        tour, feasible, cost = plan_tour(points, BUDGET)

    # update pseudoprobings for the new positions
    g = kernel_selection(Pxy, Pz)
    Nz = list(g.predict(Nxy))

    # remove (temporarily added) depot
    Nxy = Nxy[1:]
    Nz = Nz[1:]

    # evaluate fitness
    g = kernel_selection(Pxy + Nxy, Pz + Nz)
    var = evalu_var(g, TSxy)
    print(var)

# assess probing
g = gp(Pxy + Nxy, Pz + Nz)

# compare predictions to true resource levels
result = evalu_sum(g, TSxy, TSz)
print('Error=\t', fmt(result))
print()
show

# eliminate a point at random to allow the insertion of a new one
# idx = random.randrange(len(Nxy))
# Nxy.pop(idx)
# Nz.pop(idx)
# tour.remove(idx)
#
# def fu(i):
#     return i if i < idx else i - 1
#
#
# tour = list(map(fu, tour))


# Probesxy, Probesz = Probesxy + [(hx, hy)], Probesz + [f5(hx, hy)]
# Probesxy, Probesz = [], []
