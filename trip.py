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
from aux import kernel_selector, plan_tour, current_milli_time
from sklearn.gaussian_process import GaussianProcessRegressor


class Trip:
    """Handles model selection and tour calculation

    Keyword arguments:
    depot -- tuple indicating starting point of the upcoming trip
    first_xys -- list of points (tuples) already probed in previous trips
    first_zs -- list of measurements in previous trips
    """

    def __init__(self, depot, first_xys, first_zs, budget, plotter=None):
        self.depot = depot
        self.first_xys = first_xys
        self.first_zs = first_zs
        self.kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2)) #  Matern(length_scale_bounds=(0.000001, 100000), nu=1.6)
        self.xys, self.zs, self.tour = [], [], []
        self.plotter = plotter
        self.budget = budget
        self.model_time = 0
        self.tour_time = 0
        self.pred_time = 0
        self.plotvar = False

    def select_kernel(self):
        start = current_milli_time()
        self.kernel = kernel_selector(self.first_xys, self.first_zs)
        self.model_time += current_milli_time() - start

    def fit(self, kernel=None):
        start = current_milli_time()
        if kernel == None: kernel = self.kernel
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, copy_X_train=True, random_state=42)
        self.model.fit(self.first_xys, self.first_zs)
        self.model_time += current_milli_time() - start

    def calculate_tour(self):
        start = current_milli_time()
        self.tour, self.feasible, self.cost = plan_tour([self.depot] + self.xys, self.budget, exact=True)
        self.tour_time += current_milli_time() - start

    def store(self):
        self.stored_trip_xys = self.xys.copy()
        self.stored_tour = self.tour.copy()

    def restore(self):
        self.xys = self.stored_trip_xys.copy()
        self.tour = self.stored_tour.copy()

    def store2(self):
        self.stored2_trip_xys = self.xys.copy()
        self.stored2_tour = self.tour.copy()

    def restore2(self):
        self.xys = self.stored2_trip_xys.copy()
        self.tour = self.stored2_tour.copy()

    def store3(self):
        self.stored3_trip_xys = self.xys.copy()
        self.stored3_tour = self.tour.copy()

    def restore3(self):
        self.xys = self.stored3_trip_xys.copy()
        self.tour = self.stored3_tour.copy()

    def try_while_possible(self, f):
        """Apply f() while the tour is feasible."""
        self.store()
        while True:
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
        zs = [] if len(self.xys) == 0 else self.predict(self.xys)
        trip2 = Trip(self.depot, self.first_xys + self.xys, self.first_zs + list(zs), self.budget, self.plotter)
        trip2.fit(self.kernel)
        stds = trip2.predict_stds(xys)
        self.model_time += trip2.model_time
        self.pred_time += trip2.pred_time
        if self.plotvar: trip2.plot_var()
        return stds

    def set_add_maxvar_point_xys(self, add_maxvar_point_xys):
        self.add_maxvar_point_xys = add_maxvar_point_xys

    def add_maxvar_point(self):
        stds = list(self.stds_simulated(self.add_maxvar_point_xys))
        idx = stds.index(max(stds))
        self.xys.append(self.add_maxvar_point_xys[idx])

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