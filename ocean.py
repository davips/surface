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
from functions import *



f=f1

side, at_random, full_log, swarm, distortionf, exact_search, verbose = parse_args(argv)
(Pxy, Pz), (TSxy, TSz) = train_data(side, f), test_data(f)  # generate list P with points from previous probing and testing data
depot, attempts, feasible = (-0.0000001, -0.0000001), 5, True
trip = Trip(exact_search, depot, Pxy, Pz, TSxy, budget=100, debug=verbose)
plotter = Plotter('surface')

while True:
    # Add maximum amount of feasible points.
    while feasible:
        plotter.path([depot] + trip.future_xys, trip.gettour())  # plot path or surface
        ast = '\t*\t' if trip.issmallest_var() else '\t.\t'
        if full_log:
            trip2 = Trip(exact_search, depot, Pxy + trip.future_xys, Pz + probe(f, trip.future_xys), TSxy, budget=30, debug=not True)
            print(fmt(trip.getvar()) + ast, fmt(trip2.geterr_on(TSxy, TSz)) + '\t' + 'err\tlength=\t', len(trip.future_xys), (type(trip.getmodel().kernel).__name__[:12]).expandtabs(13))
        else:
            print(fmt(trip.getvar()))
        trip.add_rnd_simulatedprobe() if at_random else trip.add_maxvar_simulatedprobe()
        feasible = trip.isfeasible()
        if not feasible: trip.undo_last_simulatedprobing()

    # Find feasible distortion.
    minvar = trip.getvar()
    trip.store()
    c = 0
    while not feasible and c < attempts:
        swarm_distortion(trip) if swarm else trip.distort(distortionf)
        feasible = trip.isfeasible()
        c += 1

    # update pseudoprobings for the new positions
    trip.resimulate_probings() if feasible and trip.getvar() <= minvar else trip.restore()

# plotter.surface(f5, 30)
# plotter.surface(lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
# plotter.surface(lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)


# # whether it is dynamic (or static)
# dynamic = argv[1] == 'dyn'
# if dynamic: print("Dynamic mode!")
