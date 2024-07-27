import numpy as np
import pandas as pd
import os

from icecream import ic

if __name__ == '__main__':

    dataset_list = ['abc', 'thingi10k']
    noise_level_list = ['1e-2', '2e-3']
    method_list = [
        'digs', 'EAR', 'APSS', 'point_laplacian', 'SPR', 'nksr',
        'line_processing', 'siren', 'ours_hessian_5', 'ours_hessian_10',
        'ours_digs_5', 'ours_digs_10', 'neural_singular_hessian', 'SALD',
        'IterativePFN', 'RFEPS', 'NeurCAD'
    ]

    metrics = ['chamfer', 'hausdorff', 'f1']
    metrics_column = [
        'item', 'chamfer_mean', 'chamfer_std', 'hausdorff_mean',
        'hausdorff_std', 'f1_mean', 'f1_std'
    ]

    failure_cases = [
        '00993917_4049b13b8ff84e59b2cfc43a',
        '00992690_ed0f9f06ad21b92e7ffab606', 81762
    ]

    for dataset in dataset_list:
        for noise_level in noise_level_list:
            collection = None

            def append_collection(tag, data, collection):
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

                return collection

            for method in method_list:
                tag = f"{method}_{dataset}_{noise_level}"
                csv_path = os.path.join('output', 'metrics', f"{tag}.csv")
                data = pd.read_csv(csv_path)

                collection = append_collection(tag, data, collection)

                if 'hessian' in method:

                    for case in failure_cases:
                        # case = int(case) if dataset == 'thingi10k' else case
                        data = data[data['item'] != case]

                    collection = append_collection(f"{tag}_filter", data,
                                                   collection)

            collection.to_csv(
                os.path.join('output', 'metrics',
                             f"collect_{dataset}_{noise_level}.csv"))
