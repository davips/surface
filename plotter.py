import matplotlib.pyplot as plt
class Plotter:
    """Handles plotting

    Keyword arguments:
    """

    def __init__(self, name):
        plt.ion()
        self.fig = plt.figure(num=name)

    def path(self, points, tour):
        plot_path(self.plt, self.fig, points, tour)

    def surface(self, points, tour):
        raise NotImplementedError
    #     plot_path(self.plt, self.fig, points, tour)
