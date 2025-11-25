import os

import numpy as np
import pandas as pd

from icecream import ic


if __name__ == "__main__":
    method_list = ["capudf", "s2df", "octa"]

    metrics = ["chamfer", "hausdorff", "f1"]
    metrics_column = [
        "item",
        "chamfer_mean",
        "chamfer_std",
        "hausdorff_mean",
        "hausdorff_std",
        "f1_mean",
        "f1_std",
    ]

    collection = None

    def append_collection(tag, data, collection):
        metric_collect = [tag]
        for metric in metrics:
            metric_collect.append(data[metric].mean())
            metric_collect.append(data[metric].std())

        metrics_frame = pd.DataFrame([metric_collect], columns=metrics_column)

        if collection is None:
            collection = metrics_frame
        else:
            collection = pd.concat([collection, metrics_frame])

        return collection

    for method in method_list:
        csv_path = os.path.join("output", "metrics_srb", f"{method}.csv")
        data = pd.read_csv(csv_path)

        collection = append_collection(method, data, collection)

    collection.to_csv(os.path.join("output", "metrics", "collect_srb.csv"))
