import nibabel as nib
import pytest
from huggingface_hub import hf_hub_download

from id_uptake_values import sul, suv
from id_uptake_values.debug_plots import create_debug_image
from id_uptake_values.estimation import knn_model

REPO = "DEPICT-RH/Multimodal-HC"
SUB = "sub-000/ses-quadra"
STEM = "sub-000_ses-quadra"
CT_STEM = f"{STEM}_acq-LOWDOSE_ce-none_rec-ac"
FILES = {
    "pet": f"train/{SUB}/pet/{STEM}_trc-18FFDG_rec-acstatPSF_pet.nii.gz",
    "ct": f"train/{SUB}/ct/{CT_STEM}_ct.nii.gz",
    "totalseg": f"train/derivatives/totalsegmentator/{SUB}/ct/{CT_STEM}_seg-total_dseg.nii.gz",
    "tissueseg": f"train/derivatives/totalsegmentator/{SUB}/ct/{CT_STEM}_seg-tissue_dseg.nii.gz",
    "bodyseg": f"train/derivatives/totalsegmentator/{SUB}/ct/{CT_STEM}_seg-body_dseg.nii.gz",
}


@pytest.fixture(scope="module")
def sub000():
    return {k: hf_hub_download(REPO, f, repo_type="dataset") for k, f in FILES.items()}


def test_suv_pipeline_sub000(sub000):
    pet = nib.load(sub000["pet"])
    img, constants = suv.suv_id(pet, nib.load(sub000["ct"]), nib.load(sub000["totalseg"]), nib.load(sub000["bodyseg"]))

    assert img.shape == pet.shape
    assert constants["estimated_activity_MBq"] > 0
    assert 20 < constants["estimated_body_mass_kg"] < 250
    assert constants["estimated_suv_denominator"] > 0


def test_sul_pipeline_sub000(sub000):
    pet = nib.load(sub000["pet"])
    img, constants = sul.sul_id(pet, nib.load(sub000["ct"]), nib.load(sub000["totalseg"]),
                                nib.load(sub000["tissueseg"]), nib.load(sub000["bodyseg"]))

    assert img.shape == pet.shape
    assert constants["estimated_activity_MBq"] > 0
    assert 10 < constants["estimated_lbm_kg"] < 200
    assert constants["estimated_sul_denominator"] > 0


def test_debug_image_sub000(sub000, tmp_path):
    out = tmp_path / "debug.png"
    create_debug_image(*(nib.load(sub000[k]) for k in ("pet", "ct", "totalseg", "tissueseg", "bodyseg")), out, knn_model)
    assert out.stat().st_size > 0
