from pswarm_py import pswarm
from numpy import zeros
from numpy import ones
from aux import *


def swarm_distortion(trip):
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
