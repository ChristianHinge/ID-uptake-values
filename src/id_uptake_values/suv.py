import argparse
import json
from pathlib import Path

import nibabel as nib

from .debug_plots import create_debug_image
from .estimation import build_outputs, estimate_dose_and_volume, knn_model
from .models import KNNBodyVolume


def suv_id(pet_img, ct_img, ts_total_img, ts_body_img):
    total_dose, total_body_mass = estimate_dose_and_volume(pet_img, ct_img, ts_total_img, None, ts_body_img,
                                                           KNNBodyVolume, knn_model / "weight")
    return build_outputs(pet_img, total_dose, total_body_mass, "suv", "estimated_body_mass_kg")


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate SUV (body-mass-corrected) from PET/CT and TotalSegmentator masks.")
    parser.add_argument("--pet", required=True, type=Path, help="Path to PET image (NIfTI).")
    parser.add_argument("--ct", required=True, type=Path, help="Path to CT image (NIfTI).")
    parser.add_argument("--totalseg", required=True, type=Path, help="Path to TotalSegmentator 'total' segmentation (NIfTI).")
    parser.add_argument("--bodyseg", required=True, type=Path, help="Path to TotalSegmentator body segmentation (NIfTI).")
    parser.add_argument("--output-image", type=Path, default=None, help="Path to save the SUV image (NIfTI).")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to save the SUV constants (JSON).")
    parser.add_argument("--debug-image", type=Path, default=None, help="Path to save a debug PNG of the axial cropping and ADF.")
    args = parser.parse_args()

    if args.output_image is None and args.output_json is None and args.debug_image is None:
        parser.error("at least one of --output-image, --output-json or --debug-image must be given")

    return args


def main():
    args = parse_args()

    pet_img = nib.load(args.pet)
    ct_img = nib.load(args.ct)
    ts_total_img = nib.load(args.totalseg)
    ts_body_img = nib.load(args.bodyseg)

    if args.debug_image is not None:
        create_debug_image(pet_img, ct_img, ts_total_img, None, ts_body_img, args.debug_image, knn_model)

    if args.output_image is None and args.output_json is None:
        return

    suv_img, constants = suv_id(pet_img, ct_img, ts_total_img, ts_body_img)

    if args.output_image is not None:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        nib.save(suv_img, args.output_image)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(constants, handle, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
