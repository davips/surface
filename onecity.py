from aux import random_distortion

def onecity_distortion(trip, TSxy, nb, max_failures):
    """Distort one city at a time."""
    print('out: Distorting one city at a time...')
    trip_var_min_internal_loop = sum(trip.stds_simulated(TSxy))
    # trip_var_max = evalu_var(trip, trip.xys)
    trip.store3()
    failures = 0
    for b in range(0, nb):
        trip.distort1(random_distortion)
        trip.calculate_tour()
        # if trip.feasible: trip_var = evalu_var(trip, trip.xys)
        if trip.feasible: trip_var = sum(trip.stds_simulated(TSxy))
        # print(trip_var, trip_var_max, trip.feasible)
        if trip.feasible and trip_var < trip_var_min_internal_loop:
            # if trip.feasible and trip_var > trip_var_max:
            failures = 0
            # print('ok')
            trip_var_min_internal_loop = trip_var
            trip.store3()
            # trip.plot_path()
        else:
            failures += 1
            if failures > max_failures: break
            trip.restore3()
    trip.restore3()