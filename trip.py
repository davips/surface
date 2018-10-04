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
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, WhiteKernel
from numpy.random import uniform, randint
from plotter import Plotter
from aux import kernel_selector, plan_tour, current_milli_time, no_distortion, random_distortion, dist
from sklearn.gaussian_process import GaussianProcessRegressor


class Trip:
    """Handles model selection and tour calculation

    Keyword arguments:
    depot -- tuple indicating starting point of the upcoming trip
    first_xys -- list of points (tuples) already probed in previous trips
    first_zs -- list of measurements in previous trips
    budget -- max allowed distance + number of points
    """

    def __init__(self, depot, first_xys, first_zs, budget, plotter=None):
        self.depot = depot
        self.first_xys = first_xys
        self.first_zs = first_zs
        # self.kernel = Matern(length_scale_bounds=(0.000001, 100000), nu=2.5)
        # self.kernel = Matern(length_scale_bounds=(0.000001, 100000), nu=2.5) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
        # self.kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
        self.kernel = RationalQuadratic(length_scale_bounds=(0.08, 100))
        self.xys, self.zs, self.tour, self.cost = [], [], [], 0
        self.plotter = plotter
        self.budget = budget
        self.model_time = 0
        self.tour_time = 0
        self.pred_time = 0
        self.plotvar = False
        self.plotpred = False

    def select_kernel(self):
        start = current_milli_time()
        self.kernel = kernel_selector(self.first_xys, self.first_zs)
        self.model_time += current_milli_time() - start

    def fit(self, kernel=None, n=5):
        start = current_milli_time()
        if kernel is None: kernel = self.kernel
        # self.model = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2)), n_restarts_optimizer=10)
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n, copy_X_train=True)
        self.model.fit(self.first_xys, self.first_zs)
        self.model_time += current_milli_time() - start

    def calculate_tour(self):
        """ps. Keep the old tour if it's still within the budget and the number of cities remains compatible (the same)."""
        start = current_milli_time()
        xys = [self.depot] + self.xys
        n = len(self.tour)
        tmptour = self.tour + [0]
        self.cost, self.feasible = n, True
        for i in list(range(n)):
            a, b = xys[tmptour[i]]
            c, d = xys[tmptour[i + 1]]
            self.cost += dist(a, b, c, d)

        if self.cost > self.budget or self.tour == [] or n != len(xys):
            self.tour, self.feasible, self.cost = plan_tour([self.depot] + self.xys, self.budget, exact=True)

        self.tour_time += current_milli_time() - start

    def store(self):
        self.stored_xys = self.xys.copy()
        self.stored_tour = self.tour.copy()

    def restore(self):
        self.xys = self.stored_xys.copy()
        self.tour = self.stored_tour.copy()

    def store2(self):
        self.stored2_xys = self.xys.copy()
        self.stored2_tour = self.tour.copy()

    def restore2(self):
        self.xys = self.stored2_xys.copy()
        self.tour = self.stored2_tour.copy()

    def store3(self):
        self.stored3_xys = self.xys.copy()
        self.stored3_tour = self.tour.copy()

    def restore3(self):
        self.xys = self.stored3_xys.copy()
        self.tour = self.stored3_tour.copy()

    def add_while_possible(self, f):
        """Apply f() to add a point while the tour is feasible."""
        self.store()
        while True:
            if self.cost + 1 > self.budget:
                self.restore()
                break
            f()
            self.calculate_tour()
            if not self.feasible:
                self.restore()
                break
            self.store()

    def middle_insertion(self):
        """Add new point between two adjacent random points."""
        points = [self.depot] + self.xys
        ttt = list(zip(self.tour, self.tour[1:]))
        ida, idb = ttt[randint(len(ttt) - 1)]
        (a, b), (c, d) = points[ida], points[idb]
        m, n = (a + c) / 2, (b + d) / 2
        self.xys.append((m, n))

    def predict_stds(self, xys):
        start = current_milli_time()
        _, stds = self.model.predict(xys, return_std=True)
        self.pred_time += current_milli_time() - start
        return stds

    def predict(self, xys):
        start = current_milli_time()
        preds = self.model.predict(xys, return_std=False)
        self.pred_time += current_milli_time() - start
        return preds

    def stds_simulated(self, xys):
        """Simulate probings using predicted values, induces a model and return std deviations."""
        zs = [] if len(self.xys) == 0 else self.predict(self.xys)
        trip = Trip(self.depot, self.first_xys + self.xys, self.first_zs + list(zs), self.budget, self.plotter)
        trip.fit(self.kernel)
        stds = trip.predict_stds(xys)
        self.model_time += trip.model_time
        self.pred_time += trip.pred_time
        if self.plotvar: trip.plot_var()
        if self.plotpred: trip.plot_pred()
        return stds

    def add_maxvar_point(self, xys):
        """Intended to act as a partial function application. Return a function that appends the maxvar point (in xys) to the trip."""

        def f():
            stds = list(self.stds_simulated(xys))
            idx = stds.index(max(stds))
            self.xys.append(xys[idx])

        return f

    def add_random_point(self):
        self.xys.append((uniform(), uniform()))

    def remove_at_random(self):
        idx = randint(len(self.xys))
        del self.xys[idx]
        self.tour.remove(idx)
        for i in range(0, len(self.tour)):
            if self.tour[i] > idx: self.tour[i] -= 1

    def plot_path(self):
        if self.plotter is not None: self.plotter.path([self.depot] + self.xys, self.tour)

    def plot_var(self):
        if self.plotter is not None: self.plotter.surface(lambda x, y: self.model.predict([(x, y)], return_std=True)[1][0], 30, 0, 1)

    def plot_pred(self):
        if self.plotter is not None: self.plotter.surface(lambda x, y: self.model.predict([(x, y)])[0], 30, 0, 50)

    def fitness(self, xys, TSxy, distortf=no_distortion):
        """Return "total" variance of a given solution xys for a given test set.
           May sum up time elapsed across different threads (e.g. when used inside swarm.py).
        """
        trip = Trip(self.depot, self.first_xys, self.first_zs, self.budget)
        trip.xys = xys.copy()
        if distortf != no_distortion:
            trip.tour = self.tour.copy()  # Copy tour just to be able to call distort().
            trip.distort(distortf)
        trip.kernel = self.kernel
        trip.fit()
        trip.calculate_tour()
        v = sum(trip.stds_simulated(TSxy)) + (0 if trip.feasible else 10000 * (trip.cost - trip.budget)), trip.xys
        self.tour_time += trip.tour_time
        self.pred_time += trip.pred_time
        self.model_time += trip.model_time
        return v

    def distort(self, distortion_function):
        """Apply a custom distortion function to all points, except depot and last."""
        points = [self.depot] + self.xys
        for ida, idb, idc in zip(self.tour, self.tour[1:], self.tour[2:]):
            (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
            self.xys[idb - 1] = distortion_function(a, b, c, d, e, f)

    def distort1(self, distortion_function):
        """Apply a custom distortion function to one random point between depot and last."""
        points = [self.depot] + self.xys
        ttt = list(zip(self.tour, self.tour[1:], self.tour[2:]))
        ida, idb, idc = ttt[randint(len(ttt) - 2)]
        (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
        self.xys[idb - 1] = distortion_function(a, b, c, d, e, f)

    def distort1b(self, distortion_function):
        """Apply a custom distortion function to one random point."""
        idx = randint(len(self.xys))
        (a, b) = self.xys[idx]
        self.xys[idx] = distortion_function(a - 0.1, b, a, b, a + 0.1, b)
