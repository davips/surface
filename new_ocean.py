from sys import argv
from aux import train_data, test_data, evalu_var, evalu_sum, probe, distort1, random_distortion
from numpy.random import seed, randint
from functions import *
from plotter import Plotter
from trip import *

seed(int(argv[2]))
plot, budget, na, nb, side, f = argv[1] == 'p', 8, int(argv[3]), int(argv[4]), int(argv[5]), f5
(first_xys, first_zs), (TSxy, TSz), depot, max_failures = train_data(side, f, False), test_data(f), (-0.0000001, -0.0000001), nb / 5
plotter = Plotter('surface') if plot else None
trip = Trip(depot, first_xys, first_zs, budget, plotter)
trip.select_kernel()
trip.fit()
trip_var_min = 9999999

print('out: Adding random...')
trip.try_while_possible(trip.add_random_point)

for a in range(0, na):
    print('out: Adding neighbors...')
    trip.try_while_possible(trip.middle_insertion)

    # Distort one city at a time.
    print('out: Distorting...')
    trip_var_max = evalu_var(trip.model, trip.xys)
    new_trip_xys = trip.xys.copy()
    failures = 0
    for b in range(0, nb):
        distort1(trip.depot, trip.xys, trip.tour, random_distortion)
        trip.calculate_tour()
        if trip.feasible: trip_var = evalu_var(trip.model, trip.xys)
        if trip.feasible and trip_var > trip_var_max:
            trip_var_max = trip_var
            new_trip_xys = trip.xys.copy()
        else:
            # failures += 1
            # if failures > max_failures: break
            trip.xys = new_trip_xys.copy()

    trip.xys = new_trip_xys.copy()

    print("out: Inducing with simulated data...")
    zs = trip.model.predict(trip.xys, return_std=False)
    trip2 = Trip(depot, first_xys + trip.xys, first_zs + list(zs), budget, plotter)
    trip2.fit(trip.kernel)

    trip_var = evalu_var(trip2.model, TSxy)

    if trip_var < trip_var_min:
        trip_var_min = trip_var
        new_trip_xys2 = trip.xys.copy()
    else:
        trip.xys = new_trip_xys2.copy()

    # Logging.
    print("out: Inducing with real data to evaluate error...")
    zs = trip.model.predict(trip.xys, return_std=False)
    trip3 = Trip(depot, first_xys + trip.xys, first_zs + probe(f, trip.xys), budget, plotter)
    # trip3.kernel_selector() #TODO descomentar na versão final e tirar kernel abaixo
    trip3.fit(trip.kernel)
    error = evalu_sum(trip3.model, TSxy, TSz)
    print(trip_var_min, error, sep='\t')

    # Plotting.
    if plot:
        trip.calculate_tour()
        trip.plot()

    # Remove city at random.
    e = randint(len(trip.xys))
    del trip.xys[e]
    trip.tour.remove(e)
    for i in range(0, len(trip.tour)):
        if trip.tour[i] > e: trip.tour[i] -= 1

trip.xys = new_trip_xys2.copy()
