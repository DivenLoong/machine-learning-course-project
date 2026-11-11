"""Verify the primary local submission files against course samples and release hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

REPORT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream)
        return table.fieldnames, list(table)


def checked(actual, sample, size, header):
    actual_header, output = read(actual)
    sample_header, expected = read(sample)
    assert actual_header == sample_header == header, (actual_header, sample_header)
    assert len(output) == len(expected) == size
    keys = [r[header[0]] for r in output]
    assert len(set(keys)) == len(keys)
    assert keys == [r[header[0]] for r in expected], actual
    return output


def verify_hash(ledger, slug, path):
    expected = next(r["sha256_local"] for r in ledger if r["competition"] == slug and r["filename"] == path.name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected and actual == expected, path
    print(f"{path.name}: {actual}")


def main(base):
    plant = base / "机器学习1/2026task1"
    fer = base / "机器学习2"
    fovea = base / "机器学习3"
    drive = base / "机器学习4"
    _, ledger = read(REPORT / "data/kaggle_scores.csv")
    counts = json.loads((REPORT / "data/class_counts.json").read_text(encoding="utf-8"))

    plant_file = plant / "submission_v16_4model.csv"
    plant_rows = checked(plant_file, plant / "submission-for-task2.csv", 378, ["ID","Category"])
    assert all(r["Category"] in counts["plant"] for r in plant_rows)
    verify_hash(ledger, "2026task1", plant_file)

    for name in ("sub_v2_cns224.csv", "sub_bagged.csv"):
        fer_file = fer / "submissions" / name
        fer_rows = checked(fer_file, fer / "2026task2/submission.csv", 7065, ["ID","Emotion"])
        assert all(r["Emotion"] in set(map(str, range(6))) for r in fer_rows)
        verify_hash(ledger, "2026task2", fer_file)

    sys.path.insert(0, str(REPORT.parent / "task3/solution/src"))
    os.environ["FOVEA_DATA_DIR"] = str(fovea / "2026task3/detection")
    from common import pixels_rmse
    from validate_submission import main as validate_fovea

    assert math.isclose(pixels_rmse([[3.,4.]], [[0.,0.]]), math.sqrt(12.5))
    fovea_file = fovea / "solution/submissions/submission_v15.csv"
    validate_fovea(str(fovea_file))
    verify_hash(ledger, "2026task3", fovea_file)

    sys.path.insert(0, str(REPORT.parent / "task4"))
    from drive_vessel.rle import decode_mask, encode_mask
    snap = drive / "snapshots/best_public_0.81766_fr_official/submission_fr_official.csv"
    hdr, entries = read(snap)
    assert hdr == ["Id","Predicted"] and len(entries) == 20
    assert sorted(int(r["Id"]) for r in entries) == list(range(1,21))
    for row in entries:
        mask = decode_mask(row["Predicted"])
        assert mask.shape == (512,512) and np.isin(mask, (0, 1)).all()
        assert encode_mask(mask) == row["Predicted"]
    verify_hash(ledger, "2026task-4", snap)
    assert (drive / "submission.csv").read_bytes() == snap.read_bytes()
    print("Classification schemas, coordinate submission and RMSE, RLE and frozen hashes: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=lambda s: Path(s).resolve())
    main(parser.parse_args().workspace)
