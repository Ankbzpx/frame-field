import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic


for tag in ["00013453_056d6da46d5d4183875e63e0", "75658"]:
    sdf = np.load(f"output/{tag}.npy")
    dim = int(np.sqrt(len(sdf)))
    sdf = sdf.reshape(dim, dim)
    # Treat it as unsigned distance field
    # sdf = np.abs(sdf)

    min_bound = -0.01
    max_bound = 0.01

    min_level_set = sdf.min()
    max_level_set = sdf.max()

    while min_bound < min_level_set:
        min_bound /= 2

    while max_bound > max_level_set:
        max_bound /= 2

    min_offset = min_bound - min_level_set
    max_offset = max_level_set - max_bound

    levels = np.concatenate(
        [
            [
                min_level_set,
                min_bound - 0.2 * min_offset,
                min_bound - 0.1 * min_offset,
            ],
            np.linspace(min_bound, max_bound, 5),
            [
                max_bound + 0.1 * max_offset,
                max_bound + 0.2 * max_offset,
                max_level_set,
            ],
        ]
    )

    min_division = (min_bound - min_level_set) / (max_level_set - min_level_set)
    max_division = (max_bound - min_level_set) / (max_level_set - min_level_set)
    bound_division = (0 - min_level_set) / (max_level_set - min_level_set)

    sdf_cm = mpl.colors.LinearSegmentedColormap.from_list(
        "SDF",
        [
            (0, "#EBEAD6"),
            (min_division, "#B2A49E"),
            (bound_division, "#743E66"),
            (max_division, "#B2A49E"),
            (1, "#EBEAD6"),
        ],
        N=256,
    )

    fig = plt.figure(figsize=(10, 10))

    plt.contourf(sdf, levels=levels, cmap=sdf_cm)
    plt.contour(sdf, levels=levels, colors="#92797C", linestyles="dashed")
    plt.contour(sdf, levels=[0.0], colors="#370544")
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(
        f"output/{tag}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )
