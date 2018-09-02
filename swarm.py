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

from pswarm_py import pswarm
from numpy import zeros
from numpy import ones
from aux import *


def swarm_distortion(trip):
    def py_outf(it, leader, fx, x):
        # trip.refit2(tuplefy(x))
        # if trip.isfeasible():
        #     return 1.0  # negative number = stop
        # else:
        return 1.0

    def py_objf(xs):
        def var(x):
            trip.refit2(tuplefy(x))
            v = trip.getvar()
            return v + 10 * (trip.cost - trip.last_budget) if trip.penalize() else v  # penalize() might update cost

        return [var(x) for x in xs]

    x0 = flat(trip.future_xys)
    variabs = len(x0)
    problem = {'Variables': variabs, 'objf': py_objf, 'lb': zeros(variabs), 'ub': ones(variabs), 'x0': x0}
    # , 'A': [[-1.0 / sqrt(3), 1], [-1.0, sqrt(3)], [1.0, sqrt(3)]], 'b': [0, 0, 6]
    options = {'maxf': 3000, 'maxit': 3000, 'social': 0.5, 'cognitial': 0.5, 'fweight': 0.4
        , 'iweight': 0.9, 'size': 200, 'iprint': 10, 'tol': 1E-5, 'ddelta': 0.5, 'idelta': 2.0
        , 'outputfcn': py_outf, 'vectorized': 1}
    result = pswarm(problem, options)
    if result['ret'] == 0:  # zero means successful
        trip.refit2(tuplefy(result['x']))
