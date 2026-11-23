"""Summarize completed native/grouped pairs, rejecting mismatched inputs."""

import argparse
import json
from itertools import pairwise
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_output", type=Path)
    args = p.parse_args()
    pairs = []
    for path in sorted(args.run_output.glob("*-native/report.json")):
        other = (
            path.parent.with_name(path.parent.name.removesuffix("-native") + "-grouped")
            / "report.json"
        )
        if not other.exists():
            continue
        native, grouped = (json.loads(p.read_text()) for p in (path, other))
        for key in ("config", "sampled_hyperparameters", "data_hashes", "environment"):
            if native[key] != grouped[key]:
                raise ValueError(f"{path.parent.name}: mismatched {key}")
        intervals = {}
        for label, report in (("native", native), ("grouped", grouped)):
            boundaries = report.get("epoch_boundaries", [])
            intervals[label] = [
                right["time"] - left["time"]
                for left, right in pairwise(boundaries)
            ]
        steady = None
        if intervals["native"] and len(intervals["native"]) == len(
            intervals["grouped"]
        ):
            steady = {
                "native_epoch_intervals_seconds": intervals["native"],
                "grouped_epoch_intervals_seconds": intervals["grouped"],
                "time_reduction_percent": 100 * (
                    1 - sum(intervals["grouped"]) / sum(intervals["native"])
                ),
                "includes": (
                    "previous ensemble update, training and validation; "
                    "first epoch excluded"
                ),
            }
        pairs.append(
            {
                "case": path.parent.name.removesuffix("-native"),
                "native_seconds": native["train_wall_seconds"],
                "grouped_seconds": grouped["train_wall_seconds"],
                "time_reduction_percent": 100
                * (1 - grouped["train_wall_seconds"] / native["train_wall_seconds"]),
                "native_scores": native["scores"],
                "grouped_scores": grouped["scores"],
                "native_epochs": native["epochs_completed"],
                "grouped_epochs": grouped["epochs_completed"],
                "after_first_epoch": steady,
            }
        )
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
