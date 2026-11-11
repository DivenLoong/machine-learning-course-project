# Third-party notices

## FR-UNet

Architecture adapted from `lseventeen/FR-UNet`, MIT License.
Paper: "Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel
and Coronary Angiograph Segmentation", IEEE JBHI 2022.

## VesselSeg-Pytorch / LadderNet

LadderNet architecture adapted from `lee-zq/VesselSeg-Pytorch`, Apache License
2.0. The local implementation returns a one-channel logit map and was updated
for current PyTorch.

## Encoders

ImageNet-pretrained ResNet34 is provided through segmentation-models-pytorch.
ConvNeXt Tiny is provided through timm. Pretrained weights retain their
upstream model licenses.

## Study Group Learning

Architecture adapted from `SHI-Labs/SGL-Retinal-Vessel-Segmentation` under the
MIT License. The released DRIVE checkpoint is used as an initial model; its
training pipeline uses self-generated pseudo-labels, not test ground truth.
Paper: Zhou et al., MICCAI 2021.

## Little W-Net

Architecture adapted from `agaldran/lwnet` under the MIT License. The released
DRIVE checkpoint is used with the source paper's suggested threshold.

## External vessel datasets

- CHASE_DB1: CC BY 4.0, Kingston University Research Data Repository.
- STARE: first-observer vessel labels from the original Clemson STARE
  distribution. Check the source terms before redistributing any image.

## Additional screened implementations

- PAF / ISBI 2022 (MIT): `Limingxing00/Retinal-Vessel-Segmentation-ISBI2022`.
  Its published DRIVE prediction maps were tested at the user's request
  (public 0.75742) and were not used for the final submission.
- WA-Net (MIT):
  `dora-alvarado/wanet-retinal-vessel-segmentation`. The official DRIVE
  checkpoint was screened but not selected.
- DIAG MONAI U-Net (Apache-2.0):
  `DIAGNijmegen/drive-vessels-unet`. The official checkpoint was screened but
  not selected.
- IterNet (MIT): `conscienceli/IterNet`. Its released universal checkpoint was
  tested at the user's request (public 0.72047) and excluded from the final
  submission because the repository says all images were used for training.
- SA-UNet: `clguo/SA-UNet`. The official DRIVE Keras checkpoint was tested at
  the user's request (public 0.78976). The repository has no license file, so
  it is not part of the final submission.
- URVSM (`JRC-VPLab/URVSM`) and FSG-Net (`ZombaSY/FSG-Net-pytorch`) have
  unresolved or absent license metadata and are excluded from the final
  submission.

## Native CLAHE + Frangi U-Net recipe

The native-resolution patch-training recipe and architecture were adapted from
`LSaiko/Reti-ves-seg` under the MIT License. The competition performance was
negative and its weights are not part of the selected final submission.
