from pathlib import Path

import numpy as np

from .resampling import C1toSacrumResampler


class KNNModel:
    def __init__(self, n_neighbors, distance_function="MAE"):
        self.n_neighbors = n_neighbors
        self.distance_function = distance_function

    def load_batch(self, data):
        raise NotImplementedError("load_batch not implemented")

    def fov_sum(self, data):
        x, y = self.load_batch(data)
        return abs(x[0] - x[1]) * (y.sum())

    def load_weights(self, directory):
        self.X = np.load(Path(directory) / "X.npy")

    def _distance(self, yr, resmask):
        X = self.X[:, resmask]
        X /= X.sum(axis=-1, keepdims=True)
        yr = yr[resmask].reshape(1, -1)
        yr = yr / yr.sum()
        if self.distance_function == "MAE":
            distance = np.abs(yr - X).mean(axis=-1)
        elif self.distance_function == "MSE":
            distance = np.square(yr - X).mean(axis=-1)
        else:
            raise Exception("Unknown distance function")
        return distance

    def _resample(self, data):
        x, y = data["ts_total_x"], data["ts_total_y"]
        xmetric, ymetric = self.load_batch(data)
        resampler = C1toSacrumResampler(x, y)
        _, yr, resmask = resampler(xmetric, ymetric)
        return _, yr, resmask

    def _nn_estimate(self, distance):
        ixs = np.argsort(distance)[:self.n_neighbors]
        Xmu = self.X[ixs, :].mean(axis=0)
        Xmu = Xmu / Xmu.sum()
        return Xmu

    def predict(self, data):
        fov_sum = self.fov_sum(data)
        _, yr, resmask = self._resample(data)
        distance = self._distance(yr, resmask)
        nn_mean = self._nn_estimate(distance)
        fov_fraction = nn_mean[resmask].sum()
        return fov_sum * 1 / fov_fraction


class KNNActivity(KNNModel):
    def load_batch(self, data):
        return data["pet_x"], data["pet_y"]


class KNNLeanBodyMass(KNNModel):
    def load_batch(self, data):
        lbm = data["ts_body_y"].sum(axis=1) - 0.76 * (data["ts_tissues_y"][:, 0] + data["ts_tissues_y"][:, 1])
        lbm[lbm < 0] = 0
        return data["ts_body_x"], lbm


class KNNBodyVolume(KNNModel):
    def load_batch(self, data):
        return data["ts_body_x"], data["ts_body_y"].sum(axis=1)
