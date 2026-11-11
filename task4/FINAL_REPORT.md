# Historical task-4 report

This note records the task-4 selection and follow-up runs. The [Kaggle ledger](../report/data/kaggle_scores.csv) gives the scores for the frozen FR-UNet submission.

## Result

The best leaderboard-verified result is **0.81766 Dice** using the
MIT-licensed FR-UNet epoch-40 DRIVE checkpoint with 512x512 resizing, no TTA,
high threshold 0.65, low threshold 0.40, DTI, minimum component area 16 and FOV
masking.

The best verified Public Dice was below the 0.855 target.

## Verification and implementation changes

- Preserved the 0.81766 submission and its weight in a rollback snapshot.
- Confirmed the RLE convention and 20-row submission schema.
- Confirmed the label foreground convention from the actual files.
- Corrected FR-UNet whole-image normalization before training crops.
- Distinguished public score from non-independent local scores.
- Exposed missing ensemble metadata for DTI and low threshold.
- Added tests for RLE, metrics, model shapes and normalization order.

## Comparison runs

- Eight-way TTA: 0.79524 public.
- Native 592-pad no TTA: 0.77672 public.
- High threshold 0.80: 0.81618 public.
- Little W-Net: 0.77172 public.
- Pseudo-label FSG/FR/SGL fusion: 0.78070 public.
- Clean FR fine-tuning: 0.79157 fold-0 local.
- DIAG U-Net: 0.74878 all-training-image local.
- WA-Net: 0.78094 all-training-image local.
- Gradient-boosted pixel refiner: 0.79080 leave-one-image-out.
- IterNet universal weights: 0.72047 public.
- PAF published DRIVE prediction maps: 0.75742 public.
- SA-UNet: 0.78976 public.
- FR + SA-UNet fusion: 0.77767 public.
- FR DTI 0.65/0.50: 0.78997 public.

## Validation scope

The train and test images are the official DRIVE train/test images. Public
DRIVE checkpoints have seen every local label, so local all-20 validation
cannot estimate competition generalization. The independent checks and
leaderboard runs show that a higher offline Dice can coincide with a lower
Kaggle score. The [results summary](RESULTS.md)
separates these evaluation settings.

## Reproduction

See the [task README](README.md) for checkpoint packaging and inference.
The [evidence audit](../report/evidence_audit.md) lists the upstream weight,
packaged weight, and frozen submission hashes.

## Scores by evaluation setting

- Local all-training-image Dice: 0.78712
- Independent external probes: available for several rejected models, not a
  leaderboard estimate
- Kaggle public Dice: **0.81766**
- Kaggle private Dice: **0.82534**

## Follow-up on 2026-11-07

Five additional public submissions tested stronger or more task-matched
alternatives:

- FSG-Net DRIVE checkpoint: 0.77706
- FR threshold 0.55 without DTI: 0.77485
- Native CLAHE+Frangi U-Net at threshold 0.50: 0.75079
- Same model calibrated at threshold 0.35: 0.78848
- FR 0.75 plus native Frangi model 0.25 with DTI: 0.80897

The native U-Net reached 0.80780 Dice on the padded full-image validation
split. Its Kaggle scores used the competition's 512x512 representation.
