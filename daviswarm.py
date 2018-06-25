from pswarm_py import pswarm
from plotter import Plotter
from trip import *
from aux import *
from sys import argv
import time
from math import sqrt
from numpy import zeros
from numpy import ones

at_random = argv[1] == 'rnd'
err = argv[2] == 'err'
if at_random: print("Random mode!")
print("Printing error") if err else print("Printing 'total' variance")
(Pxy, Pz), (TSxy, TSz) = train_data(), test_data()  # generate list P with points from previous probing and testing data
depot, attempts, feasible = (-0.0000001, -0.0000001), 10, True
trip = Trip(depot, Pxy, Pz, TSxy, budget=30, debug=not True)
plotter = Plotter('surface')

while True:
    # Add maximum amount of feasible points.
    while feasible:
        plotter.path([depot] + trip.future_xys, trip.gettour())  # plot path or surface
        ast = '\t*' if trip.issmallest_var() else ''
        if err:
            trip2 = Trip(depot, Pxy + trip.future_xys, Pz + probe(trip.future_xys), TSxy, budget=30, debug=not True)
            print(fmt(trip2.geterr_on(TSxy, TSz)) + '\t' + 'err\t') #, len(trip.future_xys))
        else:
            print(fmt(trip.getvar()) + ast)  # print((type(trip.getmodel().kernel).__name__[:12]).expandtabs(13), '\ttour length=\t', len(tour), '\tvar=\t' + fmt(trip.getvar()) + ast)  # + '\tcost=\t' + fmt(cost))
        trip.add_rnd_simulatedprobe() if at_random else trip.add_maxvar_simulatedprobe()
        feasible = trip.isfeasible()
        if not feasible: trip.undo_last_simulatedprobing()

    # Find feasible distortion.
    minvar = trip.getvar()
    trip.store()
    c = 0
    while not feasible and c < attempts:
        def py_outf(it, leader, fx, x):
            trip.refit2(tuplefy(x))
            if trip.isfeasible():
                return 1.0  # negative number = stop
            else:
                return 1.0


        def py_objf(xs):
            def var(x):
                # trip.restore()
                trip.refit2(tuplefy(x))
                return trip.getvar()

            return [var(x) for x in xs]


        trip.restore()
        x0 = flat(trip.future_xys)
        variabs = len(x0)
        Problem = {'Variables': variabs, 'objf': py_objf, 'lb': zeros(variabs), 'ub': ones(variabs), 'x0': x0}
        # , 'A': [[-1.0 / sqrt(3), 1], [-1.0, sqrt(3)], [1.0, sqrt(3)]], 'b': [0, 0, 6]
        Options = {'maxf': 300, 'maxit': 300, 'social': 0.5, 'cognitial': 0.5, 'fweight': 0.4
            , 'iweight': 0.9, 'size': 30, 'iprint': 10, 'tol': 1E-5, 'ddelta': 0.5, 'idelta': 2.0
            , 'outputfcn': py_outf, 'vectorized': 1}
        result = pswarm(Problem, Options)
        if result['ret'] == 0:  # zero means successful
            # trip.restore()
            trip.refit2(tuplefy(result['x']))

        feasible = trip.isfeasible()
        print(feasible, '=fea  var (result["f"]):', result['f'])
        c += 1

    # update pseudoprobings for the new positions
    trip.resimulate_probings() if feasible and trip.getvar() <= minvar else trip.restore()

# plotter.surface(f5, 30)
# plotter.surface(lambda x, y: g.predict([(x, y)])[0], 50, 0, 50)
# plotter.surface(lambda x, y: g.predict([(x, y)], return_std=True)[1][0], 30, 0.16, 0.18)


# # whether it is dynamic (or static)
# dynamic = argv[1] == 'dyn'
# if dynamic: print("Dynamic mode!")
