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
from aux import train_data, test_data, evalu_var, evalu_sum, probe, random_distortion, current_milli_time, median_distortion
from numpy.random import seed, randint, uniform
from functions import *
from plotter import Plotter
from trip import Trip
from swarm import swarm_distortion
from ga import ga_distortion
from custom_distortion import custom_distortion, custom_distortion4

plot, seedval, time_limit, nb, side, budget, fnumber, alg = argv[1] == 'p', int(argv[2]), float(argv[3]), int(argv[4]), int(argv[5]), int(argv[6]), int(argv[7]), argv[8]
switcher = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8, 9: f9, 10: f10}
f = switcher.get(fnumber)
seed(seedval)
(first_xys, first_zs), (TSxy, TSz), depot, max_failures = train_data(side, f, False), test_data(f), (-0.00001, -0.00001), math.ceil(nb / 5)
plotter = Plotter('surface') if plot else None
trip = Trip(depot, first_xys, first_zs, budget, plotter)
print('out: selecting kernel...')
trip.select_kernel()  # TOD
print('out: fitting...')
trip.fit()
conta, acctime = 0, 0
trip_var_min = sum(trip.stds_simulated(TSxy))
while acctime < time_limit * 3600000:
    trip.tour_time, trip.model_time, trip.pred_time = 0, 0, 0
    start = current_milli_time()

    # trip.plotvar = True
    if conta > 0: trip.add_while_possible(trip.add_maxvar_point(TSxy))

    if conta == 1:
        trip_var_min = sum(trip.stds_simulated(TSxy))
    elif conta > 1:
        if alg == '1c': trip_var_min = custom_distortion(trip, TSxy, nb, random_distortion, nb / 3)
        if alg == 'sw': trip_var_min = swarm_distortion(trip, TSxy)
        if alg == 'ga': ga_distortion(trip, TSxy)
        if alg == 'a4': trip_var_min = custom_distortion4(trip, TSxy, nb, random_distortion, nb / 3)
    conta += 1

    # Logging.
    now = current_milli_time()
    print("out: Inducing with real data to evaluate error...")
    trip2 = Trip(depot, first_xys + trip.xys, first_zs + probe(f, trip.xys), budget, plotter)
    # trip2.select_kernel()
    trip2.fit(trip.kernel, 10)
    # trip2.plot_pred()
    error = evalu_sum(trip2.model, TSxy, TSz)
    total = now - start
    acctime += total
    other = total - trip.model_time - trip.pred_time - trip.tour_time
    print('res:', acctime, trip_var_min, trip_var_min, error, trip.model_time, trip.pred_time, trip.tour_time, other, total, len(trip.tour), str(trip2.kernel).replace(' ', '_'), sep='\t')

    # Plotting.
    if plot:
        trip.plot_path()
