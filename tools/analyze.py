import argparse
import os
from glob import glob

import numpy as np
from tabulate import tabulate
import openpyxl
from spft.utils import io


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    args = parser.parse_args()

    metrics = []
    for metric_path in glob(os.path.join(args.checkpoint_dir, "*", "metrics.json")):
        metric = io.load(metric_path)
        if "mean" not in metric:
            metric["mean"] = np.mean(list(metric.values()))
        metrics.append(metric)
    print(f"Loaded metrics from {len(metrics)} runs")
    s_lora_args = {} 
    for arg_path in glob(os.path.join(args.checkpoint_dir, "*", "args.json")):
        s_lora_args = io.load(arg_path)
        break
    
    
    
    metric_fns = {
        "mean-60%": lambda v: np.mean(sorted(v)[int(len(v) * 0.2) : int(len(v) * 0.8)]),
        "mean-80%": lambda v: np.mean(sorted(v)[int(len(v) * 0.1) : int(len(v) * 0.9)]),
        "mean": np.mean,
        "median": np.median,
        "min": min,
        "max": max,
        "std": np.std,
    }

    rows = []
    for name in metrics[0]:
        rows.append([name] + [metric_fns[key]([metric[name] for metric in metrics]) for key in metric_fns])
    # Generate the table as a string
    table_string = tabulate(rows, headers=[""] + list(metric_fns.keys()), tablefmt="simple_outline")
    # Print the table to the console
    print(table_string)

    # Write the table to a text file
    file_name = "".join(args.checkpoint_dir.split("/")[1:])
    with open("{}/{}_metrics.txt".format(args.checkpoint_dir, file_name), "w") as f:
        f.write(table_string)
        f.write("\n")
        #* Write out the args:
        f.write("Args:\n")
        for key, val in s_lora_args.items():
            f.write(f"{key}: {val}\n")

    #* Write table to excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metrics"
    for row in rows:
        ws.append(row)
    wb.save("{}/{}_metrics.xlsx".format(args.checkpoint_dir, file_name))

if __name__ == "__main__":
    main()
