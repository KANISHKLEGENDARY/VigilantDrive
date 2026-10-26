# Dataset record

## Contents and split counts

This folder contains a copy of the supplied training-ready YOLO dataset. The local preparation report records these retained image counts and annotation rows:

| Split | Images | Eyeclosed boxes | Eyeopen boxes | Yawn boxes |
|---|---:|---:|---:|---:|
| train | 3,188 | 2,443 | 1,133 | 518 |
| valid | 461 | 353 | 144 | 81 |
| test | 899 | 685 | 328 | 158 |

Class IDs in `data.yaml` are ordered `0: Eyeclosed`, `1: Eyeopen`, `2: Yawn`. These are object-detection box counts, not image counts per class; an image may contain several boxes.

The preparation report says blank-label samples were quarantined separately because some appeared to contain unlabeled people. It also says remaining boxes passed format/range checks but were not visually certified. The splits are reported as existing train/validation/test folders; driver/recording IDs and leakage checks have not been established.

## Provenance and license

This is a user-provided local dataset copy. Its upstream dataset URL, publisher, license, and redistribution permission have not been independently verified from the files available here. Do not redistribute it until those terms are confirmed from the source page. No ownership claim is made in this project.

## Held-out test split

The scripts default evaluation to `test`. Treat that split as held out: do not use it to tune thresholds or select checkpoints. The available preparation report identifies it as a separate folder, but does not provide source subject IDs; subject-level separation is therefore unverified.
