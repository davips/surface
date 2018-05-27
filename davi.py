from plotter import Plotter
from trip import *
from aux import *
from sys import argv
from numpy.random import normal
import time

# whether it is dynamic (or static)
dynamic = argv[1] == 'dyn'
if dynamic: print("Dynamic mode!")

# generate: list P with points from previous probing and testing data
Pxy, Pz = train_data()
TSxy, TSz = test_data()

# objects initialization
plotter = Plotter('surface')
depot=(-0.0000001, -0.0000001)
trip = Trip(depot, Pxy, Pz, budget=10)

minvar = 999999
while True:
    while trip.isfeasible():
        trip.add_maxvar_simulatedprobe()  # add point with highest variance
        if trip.isfeasible():
            tour = trip.gettour()  # store last succesful solution (a tour departing from depot)

            plotter.path([depot] + trip.future_xys, tour)  # plot path or surface
            # plotter.surface(f5, 30)
            # plotter.surface(lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
            # plotter.surface(lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)

            var = trip.gettotal_var(TSxy)
            ast = ''
            if var < minvar:
                minvar = var
                ast = '*'

            print((type(trip.getmodel().kernel).__name__[:12]).expandtabs(13), '\ttour length=\t', len(tour), '\tvar=\t' + fmt(var) + ast) # + '\tcost=\t' + fmt(cost))
            # show_path([([depot] + Nxy)[i] for i in tour], '\tvar=\t' + fmt(var) + '\terr=\t' + fmt(err) + '\tcost=\t' + fmt(cost))
        else:
            # drop last added, unfeasible, point
            trip.droplast()

    while not trip.isfeasible():
        print('distort')
        trip.distort(random_distortion)

    # update pseudoprobings for the new positions
    trip.resimulate_probing()
    # TODO: this should be done automatically on demand

    # evaluate fitness
    var = trip.gettotal_var(TSxy)
    print('var ', var)

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
