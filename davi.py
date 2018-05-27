from plotter import Plotter
from trip import *
from aux import *
from sys import argv
import time

at_random = argv[1] == 'rnd'
if at_random: print("Random mode!")

# generate: list P with points from previous probing and testing data
(Pxy, Pz), (TSxy, TSz) = train_data(), test_data()

# objects initialization
plotter = Plotter('surface')
depot, attempts = (-0.0000001, -0.0000001), 10
trip = Trip(depot, Pxy, Pz, TSxy, budget=30, debug=not True)

minvar = 999999
while True:
    while trip.isfeasible():
        tour = trip.gettour()  # store last succesful solution (a tour departing from depot)
        plotter.path([depot] + trip.future_xys, tour)  # plot path or surface
        ast = '*' if trip.issmallest_var() else ''
        print((type(trip.getmodel().kernel).__name__[:12]).expandtabs(13), '\ttour length=\t', len(tour), '\tvar=\t' + fmt(trip.getvar()) + ast)  # + '\tcost=\t' + fmt(cost))
        trip.add_rnd_simulatedprobe() if at_random else trip.add_maxvar_simulatedprobe()
        if not trip.isfeasible(): trip.undo_last_simulatedprobe()

    # Find feasible distortion.
    feasible = False
    c = 0
    while not feasible and c < attempts:
        trip.distort(random_distortion)
        feasible = trip.isfeasible()
        c += 1

    # update pseudoprobings for the new positions
    if feasible: trip.resimulate_probings()  # TODO: this should occur later automatically on demand

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


# plotter.surface(f5, 30)
# plotter.surface(lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
# plotter.surface(lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)


# # whether it is dynamic (or static)
# dynamic = argv[1] == 'dyn'
# if dynamic: print("Dynamic mode!")
