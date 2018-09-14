#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from sys import argv
from aux import train_data, test_data, evalu_var, evalu_sum, probe, random_distortion, current_milli_time
from numpy.random import seed, randint, uniform
from functions import *
from plotter import Plotter
from trip import Trip
from swarm import swarm_distortion
from ga import ga_distortion
plot, seedval, na, nb, side, budget, f = argv[1] == 'p', int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6]), f5
seed(seedval)
(first_xys, first_zs), (TSxy, TSz), depot, max_failures = train_data(side, f, False), test_data(f), (-0.00001, -0.00001), math.ceil(nb / 5)
plotter = Plotter('surface') if plot else None
trip = Trip(depot, first_xys, first_zs, budget, plotter)
trip.select_kernel_and_model()  # TOD
trip.fit()
trip_var_min = 9999999

print('out: Adding points while feasible...')
# trip.plotvar = True
trip.try_while_possible(trip.add_maxvar_point(TSxy))
# trip.try_while_possible(trip.add_random_point)

start = current_milli_time()
for a in range(0, na):
    trip.tour_time, trip.model_time, trip.pred_time = 0, 0, 0
    print('out: Adding neighbors...')
    trip.try_while_possible(trip.middle_insertion)

    ga_distortion(trip, TSxy)
    # swarm_distortion(trip, TSxy)
    # # Distort one city at a time.
    # print('out: Distorting one city at a time...')
    # trip_var_min_internal_loop = sum(trip.stds_simulated(TSxy))
    # #trip_var_max = evalu_var(trip, trip.xys)
    # trip.store3()
    # failures = 0
    # for b in range(0, nb):
    #     trip.distort1(random_distortion)
    #     trip.calculate_tour()
    #     #if trip.feasible: trip_var = evalu_var(trip, trip.xys)
    #     if trip.feasible: trip_var = sum(trip.stds_simulated(TSxy))
    #     # print(trip_var, trip_var_max, trip.feasible)
    #     if trip.feasible and trip_var < trip_var_min_internal_loop:
    #     # if trip.feasible and trip_var > trip_var_max:
    #         failures = 0
    #         #print('ok')
    #         trip_var_min_internal_loop = trip_var
    #         trip.store3()
    #         # trip.plot_path()
    #     else:
    #         failures += 1
    #         if failures > max_failures: break
    #         trip.restore3()
    # trip.restore3()

    print("out: Inducing with simulated data...")
    trip_var = sum(trip.stds_simulated(TSxy))
    if trip_var < trip_var_min:
        trip_var_min = trip_var
        trip.store2()
    else:
        trip.restore2()

    # Logging.
    print("out: Inducing with real data to evaluate error...")
    trip2 = Trip(depot, first_xys + trip.xys, first_zs + probe(f, trip.xys), budget, plotter)
    # trip2.select_kernel_and_model()
    trip2.kernel = trip.kernel
    trip2.fit()
    # trip2.plot_pred()
    error = evalu_sum(trip2.model, TSxy, TSz)
    print('res:', current_milli_time() - start, trip_var, trip_var_min, error, trip.model_time, trip.pred_time, trip.tour_time, str(trip2.kernel).replace(' ', '_'), sep='\t')

    # Plotting.
    if plot:
        trip.plot_path()

    if uniform() < 0.20:  trip.remove_at_random()

trip.restore2()
