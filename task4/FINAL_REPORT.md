# Historical task-4 report

The integrated audit later verified Private Dice 0.82534 for the frozen submission. See the course report and the archived Kaggle submission ledger for the updated record.

## Result

The best leaderboard-verified result is **0.81766 Dice** using the
MIT-licensed FR-UNet epoch-40 DRIVE checkpoint with 512x512 resizing, no TTA,
high threshold 0.65, low threshold 0.40, DTI, minimum component area 16 and FOV
masking.

This is below the requested 0.855. No compliant result above 0.81766 was
verified. I did not substitute an unverified offline estimate or use official
DRIVE test labels.

## What was fixed

- Preserved the 0.81766 submission and its weight in a rollback snapshot.
- Confirmed the RLE convention and 20-row submission schema.
- Confirmed the label foreground convention from the actual files.
- Corrected FR-UNet whole-image normalization before training crops.
- Distinguished public score from non-independent local scores.
- Exposed missing ensemble metadata for DTI and low threshold.
- Added tests for RLE, metrics, model shapes and normalization order.

## What did not work

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

## Why 0.855 remains unresolved

The train and test images are the official DRIVE train/test images. Public
DRIVE checkpoints have seen every local label, so local all-20 validation
cannot estimate competition generalization. The available independent checks
and leaderboard tests show that apparently better offline ensembles can regress
by 3.7 Dice points. Reaching 0.855 requires a stronger checkpoint trained only
on DRIVE training data, or a verified multi-model optimization campaign with
more independent data. None was available in the workspace.

The leaderboard also contains two perfect 1.00000 entries. That is inconsistent
with the best published DRIVE models and is a strong sign of leakage or metric
abuse. Those entries are not evidence for a compliant target method.

## Reproduction

```powershell
python inference.py --output submission.csv
python -m pytest -q
```

`inference.py` uses the frozen FR probability archive by default, making the
submission exactly reproducible. Pass `--probability-archive ""` to regenerate
probabilities from the copied checkpoint. `BEST_MODEL/fr_unet_drive_epoch40_packaged.pt`
is the exact copied checkpoint; SHA-256 is recorded in the snapshot.

## Final distinction

- Local all-training-image Dice: 0.78712
- Independent external probes: available for several rejected models, not a
  leaderboard estimate
- Kaggle public Dice: **0.81766**
- Kaggle private Dice: unavailable
- Claimed 0.855: not achieved and not claimed

## Follow-up on 2026-11-07

Five additional public submissions tested stronger or more task-matched
alternatives:

- FSG-Net DRIVE checkpoint: 0.77706
- FR threshold 0.55 without DTI: 0.77485
- Native CLAHE+Frangi U-Net at threshold 0.50: 0.75079
- Same model calibrated at threshold 0.35: 0.78848
- FR 0.75 plus native Frangi model 0.25 with DTI: 0.80897

The native scratch U-Net reached 0.80780 on its native-resolution FOV
validation split, but the score did not transfer to the competition's
512x512 representation. The frozen FR submission remains best and unchanged.
The daily submission quota is now exhausted. No compliant 0.855 candidate was
found.
