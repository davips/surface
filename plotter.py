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
import matplotlib.pyplot as plt
from aux import *


class Plotter:
    """Handles plotting

    Keyword arguments:
    """

    def __init__(self, name):
        plt.ion()
        self.fig = plt.figure(num=name)

    def path(self, points, tour):
        plot_path(plt, self.fig, points, tour)

    def surface(self, points, tour):
        raise NotImplementedError
    #     plot_path(self.plt, self.fig, points, tour)
