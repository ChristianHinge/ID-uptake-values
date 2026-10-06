from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .adf import make_distribution_functions, get_ct_cylindrical_mask
from .resampling import C1toSacrumResampler
from .models import KNNLeanBodyMass, KNNActivity


def _z_normalized(img, head, hip):
    k = np.arange(img.shape[2])
    z_mm = (img.affine @ np.stack([np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)]))[2]
    return (z_mm - head) / (head - hip)


def _overlay(mask, color=(1, 0, 0, 0.45)):
    out = np.zeros(mask.shape + (4,))
    out[mask] = color
    return out


def _band(img_canon, mip, head, hip, cmap, vmin, vmax, label):
    zoom = img_canon.header.get_zooms()[:3]
    return {
        "mip": mip,
        "zn": _z_normalized(img_canon, head, hip),
        "span": abs(_z_normalized(img_canon, head, hip)[-1] - _z_normalized(img_canon, head, hip)[0]),
        "t_over_z": (img_canon.shape[0] * zoom[0]) / (img_canon.shape[2] * zoom[2]),
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "label": label,
    }


def _draw_adf(ax, model, adfs, band, name, prefix, greek, scale, unit):
    x, yr, resmask = model._resample(adfs)
    distance = model._distance(yr, resmask)
    ixs = np.argsort(distance)[:model.n_neighbors]
    nn_mean = model._nn_estimate(distance)
    fov_fraction = nn_mean[resmask].sum()
    fov_sum = model.fov_sum(adfs)
    total = fov_sum / fov_fraction
    scaled_patient = yr / yr.sum() * fov_fraction
    neighbors = model.X[ixs]
    ymax = max(neighbors.max(), scaled_patient.max())

    for neighbor in neighbors:
        ax.plot(x, neighbor, color="0.5", linewidth=0.6, alpha=0.5)
    ax.plot([], [], color="0.5", linewidth=0.6, label=f"K={model.n_neighbors} nearest neighbors")
    ax.plot(x, nn_mean, color="k", linewidth=2, label="Nearest neighbors mean")
    ax.plot(x, scaled_patient, color="r", linewidth=2, label="Patient scaled to nn-mean")
    ax.fill_between(x, 0, scaled_patient, where=resmask, color="r", alpha=0.2)
    for pos, color, label in ((0, "tab:blue", "C1 (z = 0)"), (-1, "tab:green", "Sacrum (z = -1)")):
        ix = np.argmin(np.abs(x - pos))
        ax.axvline(pos, color=color, linewidth=0.8, zorder=4)
        ax.scatter([x[ix]], [scaled_patient[ix]], color=color, edgecolor="k", s=70, zorder=5, label=label)
    ax.axvline(x[resmask].min(), color="k", linestyle="--", alpha=0.5)
    ax.axvline(x[resmask].max(), color="k", linestyle="--", alpha=0.5)

    window = ax.get_window_extent()
    xrange_ = 0.75 - (-3)
    frac = band["span"] / xrange_ * (window.width / window.height) * band["t_over_z"]
    ytop = ymax / (1 - frac - 0.02)
    height = frac * ytop
    bottom = ytop - height
    ax.imshow(band["mip"], origin="lower", cmap=band["cmap"], vmin=band["vmin"], vmax=band["vmax"],
              aspect="auto", extent=[band["zn"][0], band["zn"][-1], bottom, bottom + height], zorder=3)
    if band["label"]:
        ax.text(max(band["zn"][0], -3), bottom + height + 0.01 * ytop, band["label"], fontsize=9, va="bottom")

    ax.set_xlim(-3, 0.75)
    ax.set_ylim(0, ytop)
    ax.set_title(f"{name}\n{prefix}_total = {total / scale:.1f} {unit}, FOV fraction {greek} = {fov_fraction * 100:.1f}%")
    ax.set_xlabel("Normalized axial position (z)")
    ax.set_ylabel("Volume distribution density")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.3), fontsize=8)
    return total, fov_sum, fov_fraction


def create_debug_image(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img, out_path, model_dir):
    adfs = make_distribution_functions(pet_img, ts_total_img, ts_body_img, ts_tissue_img, ct_img)
    model_dose = KNNActivity(n_neighbors=40)
    model_dose.load_weights(Path(model_dir) / "activity")
    model_volume = KNNLeanBodyMass(n_neighbors=40)
    model_volume.load_weights(Path(model_dir) / "lbm")
    sul_denominator = model_dose.predict(adfs) / model_volume.predict(adfs)

    resampler = C1toSacrumResampler(adfs["ts_total_x"], adfs["ts_total_y"])
    head, hip = resampler.head_offset, resampler.hip_offset

    sul_canon = nib.funcs.as_closest_canonical(nib.Nifti1Image(pet_img.get_fdata() / sul_denominator, affine=pet_img.affine))
    ct_canon = nib.funcs.as_closest_canonical(ct_img)
    crop_mask = resample_from_to(get_ct_cylindrical_mask(ct_img), pet_img, order=0)
    crop_arr = nib.funcs.as_closest_canonical(crop_mask).get_fdata()
    sul_arr = sul_canon.get_fdata()

    pet_band = _band(sul_canon, sul_arr.max(axis=1), head, hip, "gray_r", 0, 5, "")
    ct_mip = np.clip(ct_canon.get_fdata(), -200, 250).mean(axis=1)
    ct_band = _band(ct_canon, ct_mip, head, hip, "gray_r", ct_mip.min(), np.percentile(ct_mip, 99.9), "")

    fig, (ax_td, ax_dose, ax_lbm) = plt.subplots(1, 3, figsize=(21, 7.5))

    ax_td.imshow(sul_arr.max(axis=2).T, origin="lower", cmap="gray_r", vmin=0, vmax=5, aspect="equal")
    ax_td.imshow(_overlay((crop_arr.max(axis=2) == 0).T, color=(1, 0, 0, 0.2)), origin="lower", aspect="equal")
    ax_td.set_title("Top-down projection (SUL MIP) with cropping mask\nred = cropped by CT cylindrical mask")
    ax_td.set_xlabel("Transaxial voxel (x)")
    ax_td.set_ylabel("Transaxial voxel (y, posterior to anterior)")

    _draw_adf(ax_dose, model_dose, adfs, pet_band, "Dose ADF (PET)",
              prefix="A", greek="\u03b1", scale=1e6, unit="MBq")
    _draw_adf(ax_lbm, model_volume, adfs, ct_band, "Lean body mass ADF (CT)",
              prefix="LBM", greek="\u03b2", scale=1e3, unit="kg")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
