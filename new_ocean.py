from sys import argv
from aux import train_data, test_data, evalu_var, evalu_sum, probe, distort1, random_distortion, current_milli_time
from numpy.random import seed, randint
from functions import *
from plotter import Plotter
from trip import *

seed(int(argv[2]))
plot, budget, na, nb, side, f = argv[1] == 'p', 100, int(argv[3]), int(argv[4]), int(argv[5]), f5
(first_xys, first_zs), (TSxy, TSz), depot, max_failures = train_data(side, f, False), test_data(f), (-0.0000001, -0.0000001), nb / 5
plotter = Plotter('surface') if plot else None
trip = Trip(depot, first_xys, first_zs, budget, plotter)
trip.select_kernel()  # TOD
trip.fit()
trip_var_min = 9999999

print('out: Adding random...')
trip.set_add_maxvar_point_xys(TSxy)
trip.try_while_possible(trip.add_maxvar_point)
# trip.try_while_possible(trip.add_random_point)

start = current_milli_time()
for a in range(0, na):
    trip.tour_time, trip.model_time, trip.pred_time = 0, 0, 0
    print('out: Adding neighbors...')
    trip.try_while_possible(trip.middle_insertion)

    # Distort one city at a time.
    print('out: Distorting...')
    trip_var_max = evalu_var(trip, trip.xys)
    new_trip_xys = trip.xys.copy()
    failures = 0
    for b in range(0, nb):
        distort1(trip.depot, trip.xys, trip.tour, random_distortion)
        trip.calculate_tour()
        if trip.feasible: trip_var = evalu_var(trip, trip.xys)
        if trip.feasible and trip_var > trip_var_max:
            failures = 0
            trip_var_max = trip_var
            new_trip_xys = trip.xys.copy()
        else:
            failures += 1
            # print(failures)
            if failures > max_failures: break
            trip.xys = new_trip_xys.copy()

    trip.xys = new_trip_xys.copy()
    trip.calculate_tour()

    print("out: Inducing with simulated data...")
    trip_var = sum(trip.stds_simulated(TSxy))
    if trip_var < trip_var_min:
        trip_var_min = trip_var
        new_trip_xys2 = trip.xys.copy()
    else:
        trip.xys = new_trip_xys2.copy()

    # Logging.
    print("out: Inducing with real data to evaluate error...")
    trip3 = Trip(depot, first_xys + trip.xys, first_zs + probe(f, trip.xys), budget, plotter)
    trip3.select_kernel() # ODO descomentar na versão final e tirar kernel abaixo
    trip3.fit()
    error = evalu_sum(trip3.model, TSxy, TSz)
    print(current_milli_time() - start, trip_var, trip_var_min, error, trip.model_time, trip.pred_time, trip.tour_time, sep='\t')

    # Plotting.
    if plot:
        trip.plot()

    trip.remove_at_random()

trip.xys = new_trip_xys2.copy()
