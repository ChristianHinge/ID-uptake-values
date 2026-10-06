import numpy as np


def in_fov_volume_ml(x, y):
    dz = np.abs(np.diff(x[:2]))
    return (y.sum() * dz).item()


class C1toSacrumResampler:
    def __init__(self, xts, yts, range=[0.75, -3], sample_points=1024):
        sacrum_adf = yts[:, 25 - 1]
        c1_adf = yts[:, 50 - 1]

        if (c1_vol := in_fov_volume_ml(xts, c1_adf)) < 0.5:
            raise ValueError(f"C1 volume too small {c1_vol:.2f}ml")

        if (c1_vol := in_fov_volume_ml(xts, sacrum_adf)) < 2:
            raise ValueError(f"Sacrum volume too small {c1_vol:.2f}ml")

        self.head_offset = xts[np.argmax(c1_adf)]
        self.hip_offset = xts[np.argmax(sacrum_adf)]

        self.range = range
        self.x_interp = np.linspace(self.range[0], self.range[1], sample_points)

    def _interp(self, x, y):
        return np.interp(self.x_interp, (x - self.head_offset) / (self.head_offset - self.hip_offset), y, right=0, left=0)

    def _check_fix_data_mask(self, data_mask):
        if not data_mask.any():
            raise ValueError("data_mask is empty")

        indices = np.nonzero(data_mask)
        first = indices[0][0]
        last = indices[0][-1]
        x_probe = (self.x_interp >= -1) & (self.x_interp <= 0)
        data_mask[first:last + 1] = True

        if not data_mask[x_probe].all():
            raise ValueError("data fov does not cover C1 to sacrum")

        return data_mask

    def __call__(self, x, y):
        data_mask = y > 0
        y_interp = self._interp(x, y)
        data_mask = self._interp(x, data_mask)
        data_mask = data_mask == 1
        self._check_fix_data_mask(data_mask)
        return self.x_interp, y_interp, data_mask
