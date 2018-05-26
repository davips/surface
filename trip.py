from aux import *


class Trip:
    """Handles model selection and tour

    Keyword arguments:
    depot -- tuple indicating starting point of the upcoming trip
    first_xys -- list of points (tuples) already probed in previous trips
    first_zs -- list of measurements in previous trips
    """

    def __init__(self, first_xys, first_zs, depot, budget):
        self.depot = depot
        self.budget = budget
        self.first_xys, self.first_zs = first_xys, first_zs
        self.future_xys, self.future_zs = [], []
        self.iscache_updated_for_model = False
        self.iscache_updated_for_tour = False
        # self.iscache_updated_for_zs = False

    def refit(self, new_xys, new_zs):
        """Update all points of the future trip."""
        self.future_xys, self.future_zs = new_xys, new_zs
        self.model = kernel_selection(self.first_xys + new_xys, self.first_zs + new_zs)
        self.iscache_updated_for_tour = False
        self.iscache_updated_for_model = True

    def getmodel(self):
        if not self.iscache_updated_for_model: self.refit([], [])
        return self.model

    def add_maxvar_simulatedprobe(self):
        hxy, hz, hstd = max_var(self.getmodel())
        # if dynamic: hz = f5(hx, hy)
        # print(fmt(hx), '\t', fmt(hy), '\t', fmt(hz), '\t', hstd, '\tcurrent max variance point')
        self.future_xys, self.future_zs = self.future_xys + hxy, self.future_zs + hz
        self.iscache_updated_for_tour = False
        self.iscache_updated_for_model = False

    def calculate_tour(self):
        if not self.iscache_updated_for_tour:
            tour, self.feasible, cost = plan_tour([self.depot] + self.future_xys, self.budget)
            self.tour = tour if self.feasible else []
            self.cost = cost if self.feasible else -1
            self.iscache_updated_for_tour = True

    def resimulate_probing(self):
        self.future_zs = self.getmodel().predict(self.future_xys, return_std=False)

    def isfeasible(self):
        self.calculate_tour()
        return self.feasible

    def gettour(self):
        self.calculate_tour()
        return self.tour

    def gettotal_var(self, xys):
        evalu_var(self.getmodel(), xys)

    def droplast(self):
        self.future_xys.pop()
        self.future_zs.pop()

    def push(self):
        """Stores current set of (future) points. The set goes to a stack and can be restored later with pop()."""
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def distort(self, distortion_function):
        """Apply a custom distortion function to all points."""
        # $%&$%%#"add depot to the beginning to allow triangulation (and to correct indexes to match tour indexes)
        tour = self.tour
        points = [self.depot] + self.future_xys
        for ida, idb, idc in zip(tour, tour[1:], tour[2:]):
            (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
            self.future_xys[idb - 1] = distortion_function(a, b, c, d, e, f)
        self.iscache_updated_for_model = False
        self.iscache_updated_for_tour = False


def no_distortion(a, b, c, d, e, f):
    return c, d


def median_distortion(a, b, c, d, e, f):
    """Distortion towards median line = shortening the path."""
    m, n = (a + e) / 2, (b + f) / 2
    # offset = 0.1 * (dist(a, b, c, d) + dist(c, d, e, f) - dist(a, b, e, f))
    p = 0.01
    return c + p * (m - c), d + p * (n - d)


def random_distortion(a, b, c, d, e, f):
    s = 0.01 * (dist(a, b, c, d) + dist(c, d, e, f)) / 2
    return c + normal(scale=s), d + normal(scale=s)
