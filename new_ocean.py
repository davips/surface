from aux import *
from numpy.random import normal, uniform
from functions import *

# Induce model.
side, budget, na, nb, f = 4, 40, 100000000, 100, f5
(first_xys, first_zs), (TSxy, TSz) = train_data(side, f), test_data(f)
kernel = kernel_selector(first_xys, first_zs)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
model.fit(first_xys, first_zs)

trip_xys, trip_zs, trip_var_min, a, depot = [], [], 9999999, 0, (-0.0000001, -0.0000001)
while a < na:
    # Add maximum amount of feasible points for the given budget.
    while True:
        trip_xys += [(uniform(), uniform())]
        tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)
        if not feasible:
            trip_xys = trip_xys[:-1]
            tour = old_tour.copy()
            break
        old_tour = tour.copy()

    # Add points between neighboring cities.
    # while True:

    
    trip_var_max = evalu_var(model, trip_xys)
    new_trip_xys = trip_xys.copy()
    b = 0
    while b < nb:
        distort(depot, trip_xys, tour, random_distortion)
        trip_var = evalu_var(model, trip_xys)
        if trip_var > trip_var_max:
            trip_var_max = trip_var
            new_trip_xys = trip_xys.copy()
        else:
            trip_xys = new_trip_xys.copy()
        b += 1
    trip_xys = new_trip_xys.copy()

    
    # Induce model with simulated points.
    trip_zs = model.predict(trip_xys, return_std=False)
    # kernel = kernel_selector(first_xys + trip_xys, first_zs + list(trip_zs))
    model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
    model2.fit(first_xys + trip_xys, first_zs + list(trip_zs))

    trip_var = evalu_var(model2, TSxy)

    if trip_var < trip_var_min:
        trip_var_min = trip_var
        new_trip_xys2 = trip_xys.copy()
    else:
        trip_xys = new_trip_xys2.copy()

    
    # Logging.
    # print(trip_xys)
    # kernel = kernel_selector(first_xys + trip_xys, first_zs + probe(f, trip_xys)) #TODO descomentar na versão final
    model3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, copy_X_train=True)
    model3.fit(first_xys + trip_xys, first_zs + probe(f, trip_xys))
    error = evalu_sum(model3, TSxy, TSz)
    print(trip_var_min, error, sep='\t')

    # Remove city at random.
    random.shuffle(trip_xys)
    del trip_xys[-1]
    a += 1

trip_xys = new_trip_xys2.copy()
