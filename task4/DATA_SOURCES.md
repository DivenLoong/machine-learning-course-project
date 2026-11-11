# Data sources and scope

## Competition data

The competition supplies 20 labelled training images and 20 unlabelled test
images at 565x584 pixels, consistent with DRIVE. Only the supplied training
labels are used for supervised fitting or threshold selection. The official
DRIVE test ground truth is not downloaded or used.

## CHASE_DB1

- 28 colour retinal images with two observer annotations.
- Source: Kingston University Research Data Repository,
  <https://researchdata.kingston.ac.uk/96/>.
- License: Creative Commons Attribution 4.0.
- Local use: first-observer (`1stHO`) masks only; all images are training-only
  auxiliary data.

## STARE

- 20 images with observer annotations.
- Source: Clemson STARE vessel-network page,
  <https://cecas.clemson.edu/~ahoover/stare/probing/index.html>.
- Local use: Adam Hoover (`labels-ah`) masks only; all images are training-only
  auxiliary data.

The raw datasets are stored under `external_data/` and are not needed to run
inference from the released, packaged weights.

## Initial checkpoints

- FR-UNet DRIVE epoch-40 checkpoint: `lseventeen/FR-UNet`, MIT License.
- SGL DRIVE k=8 checkpoint:
  `SHI-Labs/SGL-Retinal-Vessel-Segmentation`, MIT License.

The SGL training procedure uses self-generated pseudo-labels. No DRIVE test
ground-truth masks are used.
