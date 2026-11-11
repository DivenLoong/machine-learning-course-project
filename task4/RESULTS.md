# Experiment results

The frozen FR-UNet submission scored 0.81766 Public Dice and 0.82534 Private Dice. The [Kaggle ledger](../report/data/kaggle_scores.csv) records both scores.

| Submission | Public Dice | Decision |
|---|---:|---|
| `submission_fr_official.csv` | **0.81766** | Frozen best |
| `submission_candidate_fr_low050.csv` | 0.78997 | Rejected |
| `submission_candidate_saunet_only.csv` | 0.78976 | Rejected |
| `submission_candidate_fr_sa_6040.csv` | 0.77767 | Rejected |
| `submission_candidate_paf_only.csv` | 0.75742 | Rejected |
| `submission_candidate_iternet_universal.csv` | 0.72047 | Rejected |
| `submission_public_fr_notta_dti080.csv` | 0.81618 | Rejected |
| `submission_fr_native.csv` | 0.79524 | Rejected |
| `submission_final_legit_iter3fuse.csv` | 0.78070 | Rejected |
| `submission_public_fr_native_notta_dti060.csv` | 0.77672 | Rejected |
| `submission_public_lwnet_fov042.csv` | 0.77172 | Rejected |
| `submission_sgl_drive_v2.csv` | 0.73528 | Rejected |
| `submission_sgl_native.csv` | 0.73184 | Rejected |

FSG-Net, SA-UNet, PAF and IterNet were also evaluated. Their Public Dice scores were below the FR-UNet submission.

The next table gives mean per-image Dice on the competition's labelled
training images. Scores from checkpoints trained on those images are
in-sample. Official DRIVE test labels were excluded.

| Candidate | Preprocessing | Fold evidence | Dice |
|---|---|---:|---:|
| ResNet34-U-Net | raw RGB | fold 0 | 0.7325 |
| ResNet34-U-Net | raw RGB | fold 1 | 0.7569 |
| ResNet34 + external data | raw RGB | fold 0 | 0.7549 |
| ConvNeXt-Tiny-U-Net | raw RGB | fold 0 | 0.7086 |
| LadderNet | green CLAHE | fold 0 | 0.7033 |
| FR-UNet, official epoch-40 weights | grayscale | all 20 | 0.7923 |
| FR-UNet + 8-way TTA/postprocess | grayscale | all 20 | 0.7933 |
| SGL DRIVE k8, no TTA | raw RGB | all 20 | 0.8083 |
| SGL DRIVE k8 + 8-way TTA | raw RGB | all 20 | 0.8117 |
| SGL + 8-way TTA + DTI | raw RGB | all 20 | **0.8120** |

External-data fine-tuning and probability averaging with FR-UNet did not
improve SGL. The earlier SGL candidate used the unmodified MIT-licensed
DRIVE checkpoint with validation thresholds of 0.60 and 0.45:

- `submission_sgl_drive_v2.csv`
- `runs/sgl_drive_final_candidate_v2/masks/`
- `runs/sgl_drive_final_candidate_v2/probabilities.npz`

Pseudolabel distillation and independent-data validation
--------------------------------------------------------

The competition train and test images were matched by exact pixels to the
official DRIVE train and test images. The lower SGL score on the public
split is an observed difference between evaluations. Official DRIVE test labels
in a third-party release archive were never read or used.

The following experiments were run after the initial snapshot:

- Corrected eight-way TTA from logit averaging to probability averaging.
- Added the official FR-UNet global normalization/min-max preprocessing path.
- Generated binary and soft pseudo labels on all 20 unlabelled test images
  from FSG/FR/SGL teachers (0.5/0.3/0.2), without using target labels.
- FR-UNet binary distillation, fold 0: 0.76764.
- FR-UNet soft distillation, fold 0: 0.76799.
- SGL soft distillation, fold 0: 0.81069.
- FIVES 240-image FR-UNet fine-tuning: 0.75523 on fold 0 and a clear
  regression on independent CHASE_DB1/STARE labels; rejected.

Independent external evaluation (mean per-image Dice, thresholds selected
separately per dataset):

| Model | CHASE_DB1 | STARE |
|---|---:|---:|
| FR-UNet DRIVE | 0.74295 | 0.75386 |
| SGL DRIVE | 0.65840 | 0.73561 |
| SGL soft-distilled | 0.70031 | 0.74755 |
| Little W-Net DRIVE | 0.74522 | 0.76598 |
| FSG-Net DRIVE | 0.76799 | 0.78151 |
| FSG + 0.15 SGL-distilled | 0.76840 | 0.78248 |

FSG had the highest external Dice among the individual teachers above. This offline candidate used
FSG/FR/SGL-soft weights 0.7/0.1/0.2 at threshold 0.425:

- `submission_final_legit_softdistill.csv`
- `runs/final_legit_softdistill/masks/`
- `runs/final_legit_softdistill/ensemble_metadata.json`

Its all-training-image Dice is 0.80278, but this includes images used by the
teacher checkpoints and is not an independent estimate. It has not been
uploaded. The smallest public score improvement over the strongest teacher
remains unverified, and the evidence does not support a claim that 0.855 is
achievable under the no-test-label rule.

Iterative SGL self-distillation
-------------------------------

Three additional rounds used teacher/student pseudo-label agreement, with
validation-fold early stopping and independent CHASE_DB1/STARE evaluation:

| SGL student | Fold-0 Dice | CHASE_DB1 | STARE |
|---|---:|---:|---:|
| Initial soft student | 0.81069 | 0.70031 | 0.74755 |
| Iteration 2 | 0.80824 | 0.73206 | 0.75008 |
| Iteration 3 | 0.80492 | 0.73601 | 0.75173 |
| Iteration 4 (stopped) | 0.80485 | 0.73551 | 0.74930 |

The fourth iteration regressed on both independent sets and was rejected.
Iteration 3 was then trained once on all 20 competition labels. Its external
results are 0.73698 on CHASE_DB1 and 0.75068 on STARE.

Iteration-3 fusion settings:

- FSG-Net: 0.75
- FR-UNet: 0.10
- SGL iteration-3 student: 0.15
- Threshold: 0.45; no connected-component filtering or DTI
- All-training-image Dice: 0.80143
- Mean external Dice under a single shared threshold: 0.77431
- Files: `submission_final_legit_iter3fuse.csv`,
  `runs/final_legit_iter3fuse/`

FSG alone had a mean external Dice of 0.77475 using separate thresholds.
The shared-threshold score above is from a different calibration protocol.
The iteration-3 fusion recorded 0.78070 Public Dice.

## 2026-11-07 independent and native-resolution follow-up

The competition submission quota was used for five public checks. None
exceeded the frozen 0.81766 submission.

| Candidate | Local evidence | Public Dice | Decision |
|---|---:|---:|---|
| FSG-Net DRIVE checkpoint | existing repository candidate | 0.77706 | Rejected |
| FR-UNet threshold 0.55, no DTI | all-20 local Dice 0.79260 | 0.77485 | Rejected |
| Native CLAHE+Frangi U-Net, threshold 0.50 | padded full-image Dice 0.80780 | 0.75079 | Rejected |
| Native CLAHE+Frangi U-Net, threshold 0.35 | bilinear-label local Dice 0.80356 | 0.78848 | Rejected |
| FR 0.75 + native Frangi U-Net 0.25, DTI | fold-0 local Dice 0.77975 | 0.80897 | Rejected |

The native U-Net follows the MIT-licensed `LSaiko/Reti-ves-seg` recipe:
CLAHE green plus Frangi channels, 64x64 FOV-centered patches, 8000 patches per
epoch, native 584x565 training, and 608x608 validation padding. The best
validation Dice was 0.80780 at epoch 15, computed over the padded full image.
The FOV is used for patch sampling. Validation on these images and the
512x512 Kaggle submission use different data and evaluation settings.

The frozen rollback remains:

- `submission.csv`
- SHA-256 `1B4415C5293E17C219B9719FA274714EE233E634DC17F511B1178E685F917FB6`
- Public Dice 0.81766

The highest verified Public Dice in these runs was 0.81766, below the 0.855 target.
