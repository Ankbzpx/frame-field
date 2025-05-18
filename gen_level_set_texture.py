import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic


parser = argparse.ArgumentParser()
parser.add_argument("vis_tag", type=str, help="Visualization tag.")
args = parser.parse_args()

tag = args.vis_tag

sdf = np.load(f"output/{tag}.npy")
dim = int(np.sqrt(len(sdf)))
sdf = sdf.reshape(dim, dim)

min_level_set = -0.035
max_level_set = 0.15
levels = np.concatenate(
    [np.linspace(min_level_set, 0, 4)[:-1], np.linspace(0, max_level_set, 12)[0:]]
)
division = np.abs(min_level_set) / (max_level_set - min_level_set)

sdf_cm = mpl.colors.LinearSegmentedColormap.from_list(
    "SDF", [(0, "#DBD7C6"), (division, "#743E66"), (1, "#DBD7C6")], N=256
)


fig = plt.figure(figsize=(10, 10))

plt.contourf(sdf, levels=levels, cmap=sdf_cm)
plt.contour(sdf, levels=levels, colors="black", linewidths=0.1)
plt.contour(sdf, levels=[0.0], colors="#370544")
plt.axis("equal")
plt.axis("off")
plt.savefig(
    f"output/{tag}.png", bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
)
