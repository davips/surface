from sys import argv
from aux import *
from numpy.random import normal, uniform
from functions import *
from random import randint
from plotter import Plotter

# Induce model.
side, budget, na, nb, f = 4, 100, 100000000, 1000, f5
(first_xys, first_zs), (TSxy, TSz) = train_data(side, f), test_data(f)
kernel = Matern(length_scale_bounds=(0.000001, 100000), nu=1.6) #TODO kernel_selector(first_xys, first_zs)
print(type(kernel))
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, copy_X_train=True, random_state=42)
model.fit(first_xys, first_zs)

trip_xys, trip_zs, trip_var_min, a, depot, plot = [], [], 9999999, 0, (-0.0000001, -0.0000001), argv[1] == 'p'
if plot: plotter = Plotter('surface')

# Add maximum amount of feasible points for the given budget.
while True:
    trip_xys.append((uniform(), uniform()))
    tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)
    if not feasible:
        trip_xys = trip_xys[:-1]
        tour = old_tour.copy()
        break
    if plot: plotter.path([depot] + trip_xys, tour)
    old_tour = tour.copy()

for a in range(0, na):
    # Add points between neighboring cities.
    old_tour = tour.copy()
    old_trip_xys = trip_xys.copy()
    while True:
        middle_insertion(depot, trip_xys, tour)
        tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)

        if not feasible:
            trip_xys = old_trip_xys.copy()
            tour = old_tour.copy()
            break
        old_trip_xys = trip_xys.copy()
        old_tour = tour.copy()

    # Distort one city at a time.
    trip_var_max = evalu_var(model, trip_xys)
    new_trip_xys = trip_xys.copy()
    for b in range(0, nb):
        distort1(depot, trip_xys, tour, random_distortion)
        tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)
        if feasible: trip_var = evalu_var(model, trip_xys)
        if feasible and trip_var > trip_var_max:
            trip_var_max = trip_var
            new_trip_xys = trip_xys.copy()
        else:
            trip_xys = new_trip_xys.copy()
        b += 1
    trip_xys = new_trip_xys.copy()

    # Induce model with simulated points.
    trip_zs = model.predict(trip_xys, return_std=False)
    # kernel = kernel_selector(first_xys + trip_xys, first_zs + list(trip_zs))
    model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, copy_X_train=True)
    model2.fit(first_xys + trip_xys, first_zs + list(trip_zs))

    trip_var = evalu_var(model2, TSxy)

    if trip_var < trip_var_min:
        trip_var_min = trip_var
        new_trip_xys2 = trip_xys.copy()
    else:
        trip_xys = new_trip_xys2.copy()

    # Logging.
    # kernel = kernel_selector(first_xys + trip_xys, first_zs + probe(f, trip_xys)) #TODO descomentar na versão final
    model3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, copy_X_train=True, random_state=42)
    model3.fit(first_xys + trip_xys, first_zs + probe(f, trip_xys))
    error = evalu_sum(model3, TSxy, TSz)
    print(trip_var_min, error, sep='\t')

    # Plotting.
    if plot:
        tour, feasible, cost = plan_tour([depot] + trip_xys, budget, exact=True)
        plotter.path([depot] + trip_xys, tour)

    # Remove city at random.
    e = randint(0, len(trip_xys) - 1)
    del trip_xys[e]
    tour.remove(e)
    for i in range(0, len(tour)):
        if tour[i] > e: tour[i] -= 1

trip_xys = new_trip_xys2.copy()
