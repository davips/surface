from aux import *
from numpy.random import normal, uniform
from functions import *

# Induce model.
side, budget, na, nb, f = 4, 10, 5, 5, f5
(first_xys, first_zs), (TSxy, TSz) = train_data(side, f), test_data(f)
kernel = kernel_selector(first_xys, first_zs)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
model.fit(first_xys, first_zs)

trip_xys, trip_zs, trip_var_min, a, depot = [], [], 9999999, 0, (-0.0000001, -0.0000001)
while a < na:

    while True:  # Add maximum amount of feasible points for the given budget.
        trip_xys += [(uniform(), uniform())]
        tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)
        if not feasible:
            trip_xys = trip_xys[:-1]
            tour = old_tour
            break
        old_tour = tour

    trip_var_max = evalu_var(model, trip_xys)
    new_trip_xys = trip_xys
    b = 0
    while b < nb:
        distort(depot, trip_xys, tour, random_distortion)
        trip_var = evalu_var(model, trip_xys)
        if trip_var > trip_var_max:
            trip_var_max = trip_var
            new_trip_xys = trip_xys
        b += 1
    trip_xys = new_trip_xys

    # Induce model with simulated points.
    trip_zs = model.predict(trip_xys, return_std=False)
    kernel = kernel_selector(first_xys + trip_xys, first_zs + list(trip_zs))  # TODO recalcular kernel incluindo pontos simulados ou manter kernel inicial?
    model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
    model2.fit(first_xys + trip_xys, first_zs + list(trip_zs))

    trip_var = evalu_var(model2, TSxy)

    # Logging.
    kernel = kernel_selector(first_xys + trip_xys, first_zs + probe(f, trip_xys))
    model3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
    model3.fit(first_xys + trip_xys, first_zs + probe(f, trip_xys))
    error = evalu_sum(TSxy, TSz)
    print(trip_var, error, sep='\t')

    if trip_var < trip_var_min:
        trip_var_min = trip_var
        new_trip_xys2 = trip_xys
    else:
        trip_xys = new_trip_xys2

    # Remove city at random.
    random.shuffle(trip_xys)
    trip_xys = trip_xys[:-1]
    a += 1

trip_xys = new_trip_xys2
