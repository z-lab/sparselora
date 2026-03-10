import argparse
import json
import os
from glob import glob

import numpy as np
from tabulate import tabulate
import openpyxl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    args = parser.parse_args()

    metrics = []
    for path in glob(os.path.join(args.checkpoint_dir, "*", "metrics.json")):
        with open(path) as f:
            metric = json.load(f)
        if "mean" not in metric:
            metric["mean"] = np.mean(list(metric.values()))
        metrics.append(metric)
    print(f"Loaded metrics from {len(metrics)} runs")

    sparselora_args = {}
    for path in glob(os.path.join(args.checkpoint_dir, "*", "sparselora_config.json")):
        with open(path) as f:
            sparselora_args = json.load(f)
        break

    metric_fns = {
        "mean-60%": lambda v: np.mean(sorted(v)[int(len(v) * 0.2): int(len(v) * 0.8)]),
        "mean-80%": lambda v: np.mean(sorted(v)[int(len(v) * 0.1): int(len(v) * 0.9)]),
        "mean": np.mean, "median": np.median,
        "min": min, "max": max, "std": np.std,
    }

    rows = []
    for name in metrics[0]:
        rows.append([name] + [metric_fns[key]([m[name] for m in metrics]) for key in metric_fns])

    table = tabulate(rows, headers=[""] + list(metric_fns.keys()), tablefmt="simple_outline")
    print(table)

    file_name = "".join(args.checkpoint_dir.split("/")[1:])
    with open(f"{args.checkpoint_dir}/{file_name}_metrics.txt", "w") as f:
        f.write(table + "\n\nConfig:\n")
        for k, v in sparselora_args.items():
            f.write(f"{k}: {v}\n")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metrics"
    for row in rows:
        ws.append(row)
    wb.save(f"{args.checkpoint_dir}/{file_name}_metrics.xlsx")


if __name__ == "__main__":
    main()
