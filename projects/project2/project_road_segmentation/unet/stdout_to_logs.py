"""
Script to convert the stdout of the utrain script to a csv log
"""

import sys
import csv

METRICS = ("loss", "acc", "val_loss", "val_acc")

def metric_value(row, metric_name):
    return float(row.split(metric_name)[1].split(":")[1].split("-")[0].replace("\n", ""))

def stdout_to_logs(stdout_fpath, csv_fpath):
    logs = []
    with open(stdout_fpath, "r") as fp:
        rows = fp.readlines()
    for row in rows:
        if not "step" in row:
            continue
        loss, acc, val_loss, val_acc = (metric_value(row, metric) for metric in METRICS)
        logs.append([loss, acc, val_loss, val_acc])
    with open(csv_fpath, 'w', newline='') as fp:
        wtr = csv.writer(fp, delimiter=',')
        wtr.writerow(["epoch","acc","loss","val_acc","val_loss"])
        for i, (loss, acc, val_loss, val_acc) in enumerate(logs):
            wtr.writerow([i, loss, acc, val_loss, val_acc])

if __name__=="__main__":
    assert len(sys.argv) == 3, "Needs two args : stdout text filepath, and logs csv filepath"
    stdout_to_logs(*sys.argv[1:])
