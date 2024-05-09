import numpy as np
import pandas as pd
import os

from icecream import ic

if __name__ == '__main__':

    dataset_list = ['abc', 'thingi10k']
    noise_level_list = ['1e-2', '2e-3']
    method_list = [
        'digs', 'neural_singular_hessian', 'APSS', 'EAR', 'point_laplacian',
        'SPR', 'nksr', 'line_processing', 'ours'
    ]

    metrics = ['chamfer', 'hausdorff', 'f1']
    metrics_column = [
        'item', 'chamfer_mean', 'chamfer_std', 'hausdorff_mean',
        'hausdorff_std', 'f1_mean', 'f1_std'
    ]
    collection = None

    for dataset in dataset_list:
        for noise_level in noise_level_list:
            for method in method_list:
                tag = f"{method}_{dataset}_{noise_level}"
                csv_path = os.path.join('output', 'metrics', f"{tag}.csv")
                data = pd.read_csv(csv_path)

                metric_collect = [tag]
                for metric in metrics:
                    metric_collect.append(data[metric].mean())
                    metric_collect.append(data[metric].std())

                metrics_frame = pd.DataFrame([metric_collect],
                                             columns=metrics_column)

                if collection is None:
                    collection = metrics_frame
                else:
                    collection = pd.concat([collection, metrics_frame])

    collection.to_csv(os.path.join('output', 'metrics', f"collect.csv"))
