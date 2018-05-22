from aux import *

BUDGET = 10


class Trip:
    """Handles model selection and tour

    Keyword arguments:
    depot -- tuple indicating starting point of the upcoming trip
    first_xys -- list of points (tuples) already probed in previous trips
    first_zs -- list of measurements in previous trips
    """

    def __init__(self, depot, first_xys, first_zs):
        self.depot = depot
        self.first_xys, self.first_zs = first_xys, first_zs
        self.g = refit([], [])

    def refit(self, new_xys, new_zs):
        self.new_xys, self.new_zs = new_xys, new_zs
        self.g = kernel_selection(first_xys + new_xys, first_zs + new_zs)

    def isfeasible(self):
        tour_, feasible, cost = plan_tour([depot] + self.new_xys, BUDGET)
        return feasible

    def addmaxvar(self):
        hxy, hz, hstd = max_var(g)
        # if dynamic: hz = f5(hx, hy)
        # print(fmt(hx), '\t', fmt(hy), '\t', fmt(hz), '\t', hstd, '\tcurrent max variance point')
        self.new_xys, self.new_zs = self.new_xys + hxy, self.new_zs + hz

    def plan_tour(self):
        tour, feasible, cost = plan_tour(points, BUDGET)

    def isfeasible(self):
