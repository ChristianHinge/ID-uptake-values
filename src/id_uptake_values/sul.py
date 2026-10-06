import argparse
import json
from pathlib import Path

import nibabel as nib

from .debug_plots import create_debug_image
from .estimation import build_outputs, estimate_dose_and_volume, knn_model
from .models import KNNLeanBodyMass


def sul_id(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img):
    total_dose, total_volume = estimate_dose_and_volume(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img,
                                                        KNNLeanBodyMass, knn_model / "lbm")
    return build_outputs(pet_img, total_dose, total_volume, "sul", "estimated_lbm_kg")


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate SUL (lean-body-mass-corrected SUV) from PET/CT and TotalSegmentator masks.")
    parser.add_argument("--pet", required=True, type=Path, help="Path to PET image (NIfTI).")
    parser.add_argument("--ct", required=True, type=Path, help="Path to CT image (NIfTI).")
    parser.add_argument("--totalseg", required=True, type=Path, help="Path to TotalSegmentator 'total' segmentation (NIfTI).")
    parser.add_argument("--tissueseg", required=True, type=Path, help="Path to TotalSegmentator tissue segmentation (NIfTI).")
    parser.add_argument("--bodyseg", required=True, type=Path, help="Path to TotalSegmentator body segmentation (NIfTI).")
    parser.add_argument("--output-image", type=Path, default=None, help="Path to save the SUL image (NIfTI).")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to save the SUL constants (JSON).")
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
    ts_tissue_img = nib.load(args.tissueseg)
    ts_body_img = nib.load(args.bodyseg)

    if args.debug_image is not None:
        create_debug_image(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img, args.debug_image, knn_model)

    if args.output_image is None and args.output_json is None:
        return

    sul_img, constants = sul_id(pet_img, ct_img, ts_total_img, ts_tissue_img, ts_body_img)

    if args.output_image is not None:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        nib.save(sul_img, args.output_image)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(constants, handle, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
