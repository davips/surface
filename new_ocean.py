from sys import argv
from aux import train_data, test_data, evalu_var, evalu_sum, probe, distort1, random_distortion, current_milli_time
from numpy.random import seed, randint
from functions import *
from plotter import Plotter
from trip import *

seed(int(argv[2]))
plot, budget, na, nb, side, f = argv[1] == 'p', int(argv[6]), int(argv[3]), int(argv[4]), int(argv[5]), f5
(first_xys, first_zs), (TSxy, TSz), depot, max_failures = train_data(side, f, False), test_data(f), (-0.0000001, -0.0000001), math.ceil(nb / 5)
plotter = Plotter('surface') if plot else None
trip = Trip(depot, first_xys, first_zs, budget, plotter)
trip.select_kernel()  # TOD
trip.fit()
trip_var_min = 9999999

print('out: Adding points while feasible...')
# trip.plotvar = True
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
    trip.store3()
    failures = 0
    for b in range(0, nb):
        distort1(trip.depot, trip.xys, trip.tour, random_distortion)
        trip.calculate_tour()
        if trip.feasible: trip_var = evalu_var(trip, trip.xys)
        if trip.feasible and trip_var > trip_var_max:
            failures = 0
            trip_var_max = trip_var
            trip.store3()
        else:
            failures += 1
            if failures > max_failures: break
            trip.restore3()
    trip.restore3()

    print("out: Inducing with simulated data...")
    trip_var = sum(trip.stds_simulated(TSxy))
    if trip_var < trip_var_min:
        trip_var_min = trip_var
        trip.store2()
    else:
        trip.restore2()

    # Logging.
    print("out: Inducing with real data to evaluate error...")
    # trip3 = Trip(depot, first_xys + trip.xys, first_zs + probe(f, trip.xys), budget, plotter)
    # # trip3.select_kernel() # TODO descomentar na versão final? e ver se precisa tirar kernel abaixo
    # trip3.fit()
    # # trip3.plot_pred()
    error = 0  # evalu_sum(trip3.model, TSxy, TSz)
    print(current_milli_time() - start, trip_var, trip_var_min, error, trip.model_time, trip.pred_time, trip.tour_time, sep='\t')

    # Plotting.
    if plot:
        trip.plot_path()

    trip.remove_at_random()

trip.restore2()
