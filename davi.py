from aux import *
from sys import argv
from numpy.random import normal
import matplotlib.pyplot as plt
import time

BUDGET = 10

# whether it is dynamic (or static)
dynamic = argv[1] == 'dyn'
if dynamic: print("Dynamic mode!")

# generate list P with points from previous probing
Pxy, Pz = train_data()
depot = -0.0000001, -0.0000001

# create empty list N for new points
Nxy, Nz = [], []
Probesxy, Probesz = [], []

# prepare plotter
plt.ion()
fig = plt.figure(num='surface')
minvar = 9999999
feasible = True
while feasible:
    # select best kernel and run GP on list P ++ N
    g = kernel_selection(Pxy + Nxy, Pz + Nz)

    #   add point h with highest variance to the list N (with simulated probing as z value)
    (hx, hy), hz, hstd = max_var(g)
    if dynamic: hz = f5(hx, hy)
    # print(fmt(hx), '\t', fmt(hy), '\t', fmt(hz), '\t', hstd, '\tcurrent max variance point')
    Nxy, Nz = Nxy + [(hx, hy)], Nz + [hz]

    # define a tour departing from depot
    points = [depot] + Nxy
    tour_, feasible, cost = plan_tour(points, BUDGET)
    # path = [points[i] for i in tour_]

    # plot path
    plotx, ploty = zip(*[points[idx] for idx in tour_])
    pl, = plt.plot(plotx, ploty, 'xb-')
    fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    time.sleep(0.5)
    pl.remove()

    # plot surfaces
    # p(plt, fig, f5, 30)
    # p(plt, fig, lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
    # p(plt, fig, lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)

    # evaluate (error sum, error max or var sum)
    TSxy, TSz = test_data()
    var = evalu_var(g, TSxy, TSz)
    err = evalu_sum(g, TSxy, TSz)
    ast = ''
    if not feasible: ast = ' unfeasible'
    if var < minvar:
        if feasible: minvar = var
        ast = ' *' + ast
    print((type(g.kernel).__name__[:12]).expandtabs(13), '\ttour length=\t', len(tour_), '\tvar=\t' + fmt(var) + ast + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))

    # show_path([([depot] + Nxy)[i] for i in tour], '\tvar=\t' + fmt(var) + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))

    if feasible:
        # store solution
        tour = tour_

        # augment list of probings
        Probesxy, Probesz = Probesxy + [(hx, hy)], Probesz + [f5(hx, hy)]

    else:
        # drop last added, unfeasible, point
        Nxy = Nxy[:-1]
        Nz = Nz[:-1]

        # add depot to the beginning to allow triangulation (and correct indexes to match tour indexes)
        Nxy = [depot] + Nxy
        Nz = [depot] + Nz

        # modify all previous points to accomodate a new one
        for ida, idb, idc in zip(tour, tour[1:], tour[2:]):
            (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
            m, n = (a + e) / 2, (b + f) / 2

            # distort each point a bit at a time

            # # no distortion
            # x, y = c, d

            #   random distortion
            s = 0.01 * (dist(a, b, c, d) + dist(c, d, e, f)) / 2
            x, y = c + normal(scale=s), d + normal(scale=s)

            # #   distortion towards median line = shortening the path
            # # offset = 0.1 * (dist(a, b, c, d) + dist(c, d, e, f) - dist(a, b, e, f))
            # p = 0.01
            # x, y = c + p * (m - c), d + p * (n - d)

            Nxy[idb] = x, y

        # update pseudoprobings for the new positions
        Nz = list(g.predict(Nxy))

        # eliminate last added point
        Nxy.pop()
        Nz.pop()



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

        # remove (temporarily added) depot
        Nxy = Nxy[1:]
        Nz = Nz[1:]

        # print("vvvvvvvvv depois")
        # g2 = gp(Pxy + Nxy, Pz + Nz)
        # var = evalu_var(g2, TSxy, TSz)
        # err = evalu_sum(g2, TSxy, TSz)
        # show_path([([depot] + Nxy)[i] for i in tour], '\tvar=\t' + fmt(var) + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))

        # force loop to recheck feasibility
        feasible = True
        # print('^^^^^^^^^^^^')

# assess probing
g = gp(Pxy + Probesxy, Pz + Probesz)

# compare predictions to true resource levels
TSxy, TSz = test_data()
result = evalu_sum(g, TSxy, TSz)
print('Error=\t', fmt(result))
print()
show
