# ID-uptake-values

Estimate SUV (body-mass normalised) and SUL (lean-body-mass normalised) PET images **without** DICOM metadata such as injected dose or patient weight. Total activity and body mass / lean body mass are inferred from the images themselves, using TotalSegmentator masks and a k-nearest-neighbour model over axial distribution functions.

## Installation

```
pip install .        # or: uv sync
```

The model weights are bundled with the package.

## Usage

Inputs are NIfTI files: PET, CT, and TotalSegmentator segmentations (`total`, `body`, and for SUL also `tissue`).

```
# SUV
python -m id_uptake_values.suv --pet pet.nii.gz --ct ct.nii.gz \
    --totalseg total.nii.gz --bodyseg body.nii.gz \
    --output-image suv.nii.gz --output-json suv.json

# SUL (optionally with a debug plot of the alignment and nearest-neighbour fit)
python -m id_uptake_values.sul --pet pet.nii.gz --ct ct.nii.gz \
    --totalseg total.nii.gz --tissueseg tissue.nii.gz --bodyseg body.nii.gz \
    --output-image sul.nii.gz --output-json sul.json --debug-image debug.png
```

The JSON contains the estimated activity (MBq), the estimated body mass or lean body mass (kg), and the denominator the PET image was divided by.

The scan must cover the region from C1 to the sacrum; otherwise a `ValueError` is raised.

## Development

```
uv sync
uv run pytest
```

The tests download subject `sub-000` from the [DEPICT-RH/Multimodal-HC](https://huggingface.co/datasets/DEPICT-RH/Multimodal-HC) dataset into the Hugging Face cache.
