from pathlib import Path

import nibabel as nib

from .adf import make_distribution_functions
from .models import KNNActivity

knn_model = Path(__file__).parent / "weights/knn-melanoma-fdg"


def estimate_dose_and_volume(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img, volume_model, volume_dir):
    adfs = make_distribution_functions(pet_img, ts_total_img, ts_body_img, ts_tissue_img, ct_img)
    model_dose = KNNActivity(n_neighbors=40)
    model_dose.load_weights(knn_model / "activity")
    model_volume = volume_model(n_neighbors=40)
    model_volume.load_weights(volume_dir)
    return model_dose.predict(adfs), model_volume.predict(adfs)


def build_outputs(pet_img, total_dose, total_volume, quantity, volume_key):
    denominator = total_dose / total_volume
    constants = {
        f"estimated_{quantity}_denominator": float(denominator),
        "estimated_activity_MBq": float(total_dose) / 1e6,
        volume_key: float(total_volume) / 1e3,
    }
    img = nib.Nifti1Image(pet_img.get_fdata() / denominator, affine=pet_img.affine)
    return img, constants
