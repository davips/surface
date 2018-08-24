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
from plotter import Plotter
from trip import *
from aux import *
from sys import argv
import time
from math import sqrt
from args import *
from swarm import *

show, f, side, at_random, full_log, swarm, distortionf, exact_search, verbose = parse_args(argv)
(Pxy, Pz), (TSxy, TSz) = train_data(side, f), test_data(f)  # generate list P with points from previous probing and testing data
depot, attempts = (-0.0000001, -0.0000001), 200
trip = Trip(exact_search, depot, Pxy, Pz, TSxy, debug=verbose)
if show != 'none': plotter = Plotter('surface')

for budget in range(10, 200, 5):
    # Add maximum amount of feasible points for the given budget.
    feasible = True
    while feasible:
        trip.add_rnd_simulatedprobe() if at_random else trip.add_maxvar_simulatedprobe()
        feasible = trip.isfeasible(budget)
        if not feasible: trip.undo_last_simulatedprobing()

    # Find feasible distortion.
    minvar = trip.getvar()
    trip.store()
    if swarm:
        while not feasible:
            trip.restore()
            swarm_distortion(trip)
            feasible = trip.isfeasible(budget)
            trip.resimulate_probings()
            # log(fmt(trip.getvar()) + '\tswarm var; feasible:\t' + str(feasible))
    else:
        c = 0
        min_var = trip.getvar()
        trip.store()
        while c < attempts:
            trip.distort(distortionf)
            feasible = trip.isfeasible(budget)
            if feasible:
                trip.resimulate_probings()
                var = trip.getvar()
                # log(fmt(var) + '\tdistortion var; feasible:\t' + str(feasible))
                if var < min_var:
                    min_var = var
                    trip.store()
            c += 1
        trip.restore()

    if trip.getcost(budget) > budget: log("cost > budget!")

    # Plotting.
    if show == 'var': plotter.surface(lambda x, y: trip.getmodel().predict([(x, y)], return_std=True)[1][0], 30, 0, 1)
    if show == 'path': plotter.path([depot] + trip.future_xys, trip.gettour(budget))
    if show == 'fun': plotter.surface(lambda x, y: f(x, y), 30, 0, 50)
    if show == 'est':  # estimated function after performing all probings
        trip2 = Trip(exact_search, depot, Pxy + trip.future_xys, Pz + probe(f, trip.future_xys), TSxy, debug=not True)
        plotter.surface(lambda x, y: trip2.getmodel().predict([(x, y)])[0], 30, 0, 50)

    # Logging.
    if full_log:  # calculate error after all probings and rechoosing kernel
        trip2 = Trip(exact_search, depot, Pxy + trip.future_xys, Pz + probe(f, trip.future_xys), TSxy, debug=not True)
        print('out:\t' + fmt(trip.getvar()), fmt(trip2.geterr_on(TSxy, TSz)) + '\t' + 'err\tlength=\t', len(trip.future_xys) , '\t' , (type(trip.getmodel().kernel).__name__[:12]).expandtabs(13))
    else:
        print('out:\t' + fmt(trip.getvar()) + '\tvar')
