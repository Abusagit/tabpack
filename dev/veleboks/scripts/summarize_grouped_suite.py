"""Summarize completed native/grouped pairs, rejecting mismatched inputs."""

import argparse
import json
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
            }
        )
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
